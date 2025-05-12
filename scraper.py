import requests
from bs4 import BeautifulSoup
import re, json
from utils import bcolors
import logging
from time import sleep
import datetime

ROOT_LINK="https://www.mountainproject.com/area/105744222/boulder-canyon"
http_session = requests.Session()
total_routes = 0

def main():
    dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.f%z")
    logging.basicConfig(filename=f"./logs/{dt}_scrape.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    routes = []
    routes = get_areas(http_session.get(ROOT_LINK), ROOT_LINK)
    with open("data/routes.json", "w") as fp:
        json.dump(routes, fp)


def get_areas(page, link):
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
        tries = 0
        timeout = 0.1
        while tries <= 6:
            sub_page = http_session.get(sub_link)
            if sub_page.status_code == 200:
                 break
            logging.warning("GET %s failed with code %s", sub_link, sub_page.status_code)
            sleep(timeout)
            timeout = timeout*2
            ties += 1 
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

    routes_table = soup.find("table", id="left-nav-route-table").find_all("a")
    if routes_table == None:
        logging.error("Could not find route table for %s", link)
        return [] 

    all_route_info = []
    for location in routes_table:
#        if atag == None:
 #           logging.error("Could not find link for an unknown route in %s", link)
  #          continue

        sub_link = location["href"]

        tries = 0
        timeout = 0
        while tries <= 6:
            sub_page = http_session.get(sub_link)
            if sub_page.status_code == 200:
                break
            logging.warning("GET %s failed with code %s", sub_link, sub_page.status_code)
            sleep(timeout)
            timeout = timeout*2
            tries+= 1
        if tries >= 6:
            logging.error("GET %s exceeded number of retries", sub_link)
            return []

        all_route_info.append(get_route_info(sub_page, sub_link))

    return all_route_info


def get_route_info(page, link):
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

    route_info = {}
    route_info["avg_rating"] = rating[0]
    route_info["rating_count"] = rating[1]
    route_info["name"] = name
    route_info["grade"] = grade
    route_info["link"] = link

    return route_info


if __name__ == "__main__":
    main()
