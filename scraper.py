import requests
from bs4 import BeautifulSoup

def main():
    routes = []
    routes = get_areas(requests.get("https://www.mountainproject.com/area/113745735/cross-mountain-boulders"))
    print(routes)

def get_areas(page):
    soup = BeautifulSoup(page.content, "html.parser")
    sidebar = soup.find("div", class_="mp-sidebar")
    areas_table = sidebar.find_all("div", class_="lef-nav-row")

    area_links = []
    for arearow in areas_table:
        area_links.append(arearow.find("a")["href"])

    routes = []
    for link in area_links:
        sub_page = requests.get(link)
        soup = BeautifulSoup(sub_page.content, "html.parser")
        route_table = soup.find("table", id="left-nav-route-table")
        
        if route_table == None:
            routes.extend(get_areas(sub_page))
        else:
            routes.extend(get_routes(sub_page))


    return routes

def get_routes(page):
    soup = BeautifulSoup(page.content, "html.parser")

    routes_table = soup.find("table", id="left-nav-route-table")
    if routes_table == None:
        return [] 
    
    route_links = []
    route_names = []
    for location in routes_table:
        atag = location.find("a")
        if atag == -1 or atag == None:
            continue
        
        route_links.append(atag["href"])
        route_names.append(atag.text)
    return route_names

if __name__ == "__main__":
    main()