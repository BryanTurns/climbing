import argparse
import concurrent.futures
import datetime
import json
import logging
import re
import signal
import sys
from pathlib import Path
from time import sleep

import requests
from bs4 import BeautifulSoup

http_session = requests.Session()

total_routes = 0

# Initialized in main() once we know the requested thread pool size.
executor = None

# Routes scraped so far. Populated by get_route_info as each route is
# fetched so that whatever we have can be persisted if the scrape is
# interrupted (Ctrl-C, SIGTERM, unhandled exception) before it finishes.
all_routes = []
# Route areas (the leaf areas that contain routes) scraped so far. Each
# entry has the area's name, link, GPS coordinates, and page views.
all_route_areas = []
# Flipped by the signal handler so worker functions can bail out early.
interrupted = False

DEFAULT_THREADS = 10
DEFAULT_OUTPUT_FILE = "data.json"


def _positive_int(value):
    """argparse `type` for thread pool size: must be an int >= 1."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer (got {value!r})")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {n})")
    return n


def _parse_args(argv):
    """Parse CLI args. Returns (filename, root_link, thread_pool_size)."""
    parser = argparse.ArgumentParser(
        prog="scraper.py",
        description="Scrape Mountain Project routes under a given area.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=(
            "output filename, used as-is (no extension is appended); "
            f"written under ./data/ (default: {DEFAULT_OUTPUT_FILE})"
        ),
    )
    parser.add_argument(
        "-l",
        "--root-link",
        required=True,
        help="root Mountain Project area URL to start scraping from",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=_positive_int,
        default=DEFAULT_THREADS,
        help=f"size of the thread pool (default: {DEFAULT_THREADS})",
    )
    args = parser.parse_args(argv)
    return args.output_file, args.root_link, args.threads


def main():
    global executor

    fname, root_link, threads = _parse_args(sys.argv[1:])
    fname = Path(fname)

    # Match the HTTP connection pool to the thread pool so workers don't
    # contend over a fixed-size connection pool when -t is large.
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=threads)
    http_session.mount("https://", adapter)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)

    dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.f%z")
    logging.basicConfig(
        filename=f"./logs/{dt}_scrape.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Trigger a graceful shutdown on Ctrl-C / SIGTERM so we still write the
    # partial results to disk instead of losing the work-in-progress.
    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    try:
        get_areas(http_session.get(root_link), root_link)
    except KeyboardInterrupt:
        # If the interrupt lands in pure-Python code the signal handler may
        # not have flipped the flag yet -- treat it the same and fall
        # through to the save in `finally`.
        logging.warning("Scrape interrupted by user (KeyboardInterrupt)")
    finally:
        # Cancel queued tasks; let in-flight ones finish so their results
        # make it into all_routes before we serialize.
        executor.shutdown(wait=True, cancel_futures=True)
        _save_routes(fname)


def _handle_interrupt(signum, frame):
    global interrupted
    if interrupted:
        # Second signal: restore the default handler so a third press
        # hard-kills the process in case graceful shutdown is hanging.
        signal.signal(signum, signal.SIG_DFL)
        return
    interrupted = True
    print(
        f"\nReceived signal {signum} - finishing in-flight requests and saving partial data..."
    )
    logging.warning("Received signal %s - shutting down gracefully", signum)


def _save_routes(fname):
    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    output = {
        "route_areas": all_route_areas,
        "routes": all_routes,
    }
    with open(out_path, "w") as fp:
        json.dump(output, fp)
    msg = (
        f"Saved {len(all_routes)} routes across {len(all_route_areas)} "
        f"route areas to {out_path}"
    )
    print(msg)
    logging.info(msg)


def _parse_description_details(soup):
    """Pull every field we care about out of the ``description-details``
    table that Mountain Project renders on both area and route pages.

    Returns a dict with the union of fields seen on areas and routes:

    * ``gps``: ``{"lat": float, "lon": float}`` or ``None``
    * ``page_views_total`` / ``page_views_per_month``: ``int`` or ``None``
    * ``shared_on``: ``str`` like ``"Dec 31, 2000"`` or ``None``
    * ``elevation_ft`` / ``elevation_m``: ``int`` or ``None`` (areas only)
    * ``type``: list of climbing-style strings (``["Sport", "TR"]``) or
      ``None`` (routes only)
    * ``height_ft`` / ``height_m``: ``int`` or ``None`` (routes only)

    Missing fields stay ``None`` so a partial scrape still succeeds; the
    caller picks the keys it needs.
    """
    metadata = {
        "gps": None,
        "page_views_total": None,
        "page_views_per_month": None,
        "shared_on": None,
        "elevation_ft": None,
        "elevation_m": None,
        "type": None,
        "height_ft": None,
        "height_m": None,
    }
    table = soup.find("table", class_="description-details")
    if table is None:
        return metadata

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        label = tds[0].get_text(strip=True)
        value_text = tds[1].get_text(separator=" ", strip=True)

        if label.startswith("GPS"):
            m = re.search(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", value_text)
            if m:
                metadata["gps"] = {
                    "lat": float(m.group(1)),
                    "lon": float(m.group(2)),
                }
        elif label.startswith("Page Views"):
            # Format: "91,038 total · 295/month"
            m = re.search(
                r"([\d,]+)\s*total\D+([\d,]+)\s*/\s*month", value_text
            )
            if m:
                metadata["page_views_total"] = int(m.group(1).replace(",", ""))
                metadata["page_views_per_month"] = int(
                    m.group(2).replace(",", "")
                )
        elif label.startswith("Shared By"):
            # Format: "<username> on Dec 31, 2000" (or full month name).
            m = re.search(
                r"on\s+([A-Z][a-z]{2,9}\s+\d{1,2},\s+\d{4})", value_text
            )
            if m:
                metadata["shared_on"] = m.group(1)
        elif label.startswith("Elevation"):
            # Layout: <td>Elevation:</td><td class="imperial">7,650 ft</td>
            #         <td class="metric">2,332 m</td>
            ft_match = re.search(r"([\d,]+)", value_text)
            if ft_match:
                metadata["elevation_ft"] = int(
                    ft_match.group(1).replace(",", "")
                )
            if len(tds) >= 3:
                metric_text = tds[2].get_text(separator=" ", strip=True)
                m_match = re.search(r"([\d,]+)", metric_text)
                if m_match:
                    metadata["elevation_m"] = int(
                        m_match.group(1).replace(",", "")
                    )
        elif label.startswith("Type"):
            # Format examples:
            #   "Sport, TR, 50 ft (15 m) Fixed Hardware (4)"
            #   "Trad, 5 pitches, 600 ft (182 m)"
            #   "Boulder"  (no height)
            # The height pattern anchors the split: anything before it is a
            # comma-separated list of climbing styles, anything after is
            # decoration (Fixed Hardware link, etc.) we ignore.
            height_match = re.search(
                r"(\d[\d,]*)\s*ft\s*\(\s*(\d[\d,]*)\s*m\s*\)", value_text
            )
            if height_match:
                metadata["height_ft"] = int(
                    height_match.group(1).replace(",", "")
                )
                metadata["height_m"] = int(
                    height_match.group(2).replace(",", "")
                )
                types_text = value_text[: height_match.start()]
            else:
                types_text = value_text

            types = []
            for part in types_text.split(","):
                part = part.strip()
                if not part:
                    continue
                # Tokens with digits are pitch counts ("5 pitches"), boulder
                # grades ("V0"), aid ratings ("A1"), etc. -- not styles.
                if any(ch.isdigit() for ch in part):
                    continue
                types.append(part)
            if types:
                metadata["type"] = types
    return metadata


def get_areas(page, link):
    if interrupted:
        return []
    soup = BeautifulSoup(page.content, "html.parser")

    area_tag = soup.find("h1")
    area_name = ""
    if area_tag != None:
        area_name = area_tag.text.replace("\n", "")
        area_name = " ".join(area_name.split())
        area_name = area_name.strip(" \n\t")
    else:
        logging.warning("Could not determine area name for %s", link)
    print(f"Processing Area: {area_name}")

    sidebar = soup.find("div", class_="mp-sidebar")
    if sidebar == None:
        logging.warning("Could not locate a sidebar for %s", link)
        return []
    areas_table = sidebar.find_all("div", class_="lef-nav-row")
    if areas_table == None:
        logging.warning("Could not locate an areas table for %s", link)
        return []

    area_links = []
    for arearow in areas_table:
        atag = arearow.find("a")
        if atag == None:
            logging.error("Could not locate a link for an unknown area in %s", link)
            continue

        area_links.append(atag["href"])

    routes = []
    for sub_link in area_links:
        if interrupted:
            logging.warning("Interrupt flag set - aborting area iteration in %s", link)
            break
        tries = 0
        timeout = 0.1
        while tries <= 6:
            sub_page = http_session.get(sub_link)
            if sub_page.status_code == 200:
                break
            logging.warning(
                "GET %s failed with code %s", sub_link, sub_page.status_code
            )
            sleep(timeout)
            timeout = timeout * 2
            tries += 1
        if tries >= 6:
            logging.error("GET %s exceeded number of retries", sub_link)
            return []

        soup = BeautifulSoup(sub_page.content, "html.parser")
        route_table = soup.find("table", id="left-nav-route-table")
        if route_table == None:
            routes.extend(get_areas(sub_page, sub_link))
        else:
            routes.extend(get_routes(sub_page, sub_link))

    return routes


def get_routes(page, link):
    if interrupted:
        return []
    soup = BeautifulSoup(page.content, "html.parser")

    area_tag = soup.find("h1")
    area_name = ""
    if area_tag != None:
        area_name = area_tag.text.replace("\n", "")
        area_name = " ".join(area_name.split())
        area_name = area_name.strip(" \n\t")
    else:
        logging.warning("Could not determine area name for %s", link)
    print(f"Processing Route Area: {area_name}")

    metadata = _parse_description_details(soup)
    if metadata["gps"] is None:
        logging.warning("Could not parse GPS for route area %s", link)
    if metadata["page_views_per_month"] is None:
        logging.warning("Could not parse Page Views for route area %s", link)
    if metadata["shared_on"] is None:
        logging.warning("Could not parse Shared By date for route area %s", link)
    if metadata["elevation_ft"] is None:
        logging.warning("Could not parse Elevation for route area %s", link)

    all_route_areas.append(
        {
            "name": area_name,
            "link": link,
            "gps": metadata["gps"],
            "page_views_total": metadata["page_views_total"],
            "page_views_per_month": metadata["page_views_per_month"],
            "shared_on": metadata["shared_on"],
            "elevation_ft": metadata["elevation_ft"],
            "elevation_m": metadata["elevation_m"],
        }
    )

    routes_table = soup.find("table", id="left-nav-route-table").find_all("a")
    if routes_table == None:
        logging.error("Could not find route table for %s", link)
        return []

    all_route_info = []
    futures = []
    for location in routes_table:
        if interrupted:
            break
        sub_link = location["href"]

        futures.append(
            executor.submit(get_route_info, sub_link, area_name, link)
        )

    for future in futures:
        if interrupted:
            # Cancel anything that hasn't started yet; running tasks will
            # finish on their own and append to the global all_routes.
            future.cancel()
            continue
        try:
            route_info = future.result()
        except Exception:
            logging.exception("Route fetch failed in %s", link)
            continue
        if route_info != None:
            all_route_info.append(route_info)
    return all_route_info


def get_route_info(link, route_area_name, route_area_link):
    if interrupted:
        return
    tries = 0
    timeout = 1
    while tries <= 6:
        page = http_session.get(link)
        if page.status_code == 200:
            break
        logging.warning("GET %s failed with code %s", link, page.status_code)
        sleep(timeout)
        timeout = timeout * 2
        tries += 1
    if tries >= 6:
        logging.error("GET %s exceeded number of retries", link)
        return

    soup = BeautifulSoup(page.content, "html.parser")

    name_tag = soup.find("h1")
    name = ""
    if name_tag != None:
        name = name_tag.text.strip(" \n")
    else:
        logging.warning("Could not find name for route %s", link)

    rating_tag = soup.find("span", class_="scoreStars")
    rating = ""
    if rating_tag != None:
        rating = rating_tag.parent.text
        rating = re.findall(r"(\d*\.?\d+)", rating)
    else:
        logging.warning("Could not find rating for route %s", link)

    grade_tag = soup.find("span", class_="rateYDS")
    grade = ""
    if grade_tag != None:
        grade = grade_tag.text
        grade = re.search(r"[^Y][^D][^S]", grade)
        if grade != None:
            grade = grade.group().strip()
        else:
            logging.warning("Could not parse grade for route %s", link)
    else:
        logging.warning("Could not find grade for route %s", link)

    metadata = _parse_description_details(soup)
    if metadata["page_views_per_month"] is None:
        logging.warning("Could not parse Page Views for route %s", link)
    if metadata["shared_on"] is None:
        logging.warning("Could not parse Shared By date for route %s", link)
    if metadata["type"] is None:
        logging.warning("Could not parse Type for route %s", link)

    route_info = {}
    route_info["avg_rating"] = rating[0]
    route_info["rating_count"] = rating[1]
    route_info["name"] = name
    route_info["grade"] = grade
    route_info["link"] = link
    route_info["route_area"] = route_area_name
    route_info["route_area_link"] = route_area_link
    route_info["type"] = metadata["type"]
    route_info["height_ft"] = metadata["height_ft"]
    route_info["height_m"] = metadata["height_m"]
    route_info["page_views_total"] = metadata["page_views_total"]
    route_info["page_views_per_month"] = metadata["page_views_per_month"]
    route_info["shared_on"] = metadata["shared_on"]

    # Append to the module-level list so the route is captured even if the
    # scrape is interrupted before we get back to main().
    all_routes.append(route_info)
    return route_info


if __name__ == "__main__":
    main()
