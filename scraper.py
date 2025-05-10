import requests
from bs4 import BeautifulSoup

def main():
    links=[]
    # links.append(get_routes("https://www.mountainproject.com/area/123696872/broomfield-boulders"))
    print(links)

    areas_co = get_areas("https://www.mountainproject.com/area/105708956/colorado")
    print(areas_co)
    # for area in areas_co:
    #     print(get_routes(area))

def get_areas(base_url):
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, "html.parser")
    sidebar = soup.find("div", class_="mp-sidebar")
    areas_table = sidebar.find_all("div", class_="lef-nav-row")

    area_links = []
    for arearow in areas_table:
        area_links.append(arearow.find("a")["href"])

    
    # for location in areas_table:
    #     area_links.append(location["href"])

    for link in areas_table:
        sub_page = requests.get(link)
        


    return area_links

def get_routes(base_url):
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, "html.parser")

    routes_table = soup.find_all("table", id="left-nav-route-table")[0]

    route_links = []
    route_names = []
    for location in routes_table:
        atag = location.find("a")
        if atag == -1 or atag == None:
            continue
        
        route_links.append(atag["href"])
        route_names.append(atag.text)
    print(route_names)
    return route_links

if __name__ == "__main__":
    main()