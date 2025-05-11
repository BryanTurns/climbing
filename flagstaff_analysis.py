import matplotlib.pyplot as plt
import pandas as pd
import json
from cleaning_utils import bouldering_only, translate_boulder_grade

def main():
    with open("routes.json") as fp:
        routes = json.load(fp)

    routes_df = pd.DataFrame(routes)
    routes_df=clean_flagstaff(routes_df)
    
    plt.hist(routes_df["avg_rating"], bins=10)
    plt.savefig("fig.png")
    print(routes_df["translated_grade"].unique())


def clean_flagstaff(routes_df):
    routes_df["rating_count"] = pd.to_numeric(routes_df["rating_count"])
    routes_df["avg_rating"] = pd.to_numeric(routes_df["avg_rating"])

    routes_df = routes_df[routes_df["rating_count"] >= 5]
    
    routes_df = bouldering_only(routes_df)
    routes_df = translate_boulder_grade(routes_df)     
    return routes_df


if __name__ == "__main__":
    main()
