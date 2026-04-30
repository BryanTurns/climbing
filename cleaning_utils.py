import pandas as pd
import re


def general_clean(routes_df):
    routes_df["avg_rating"] = pd.to_numeric(routes_df["avg_rating"])
    routes_df["rating_count"] = pd.to_numeric(routes_df["rating_count"])

    routes_df = routes_df[routes_df["rating_count"] > 5]

    return routes_df

def ropes_only(routes_df):
    return routes_df.dropna(subset=["yds_grade"])

def bouldering_only(routes_df):
    # A route can carry both yds_grade and boulder_grade (a sport climb
    # that's also a boulder problem); we want it included here as long
    # as a V grade is present, regardless of whether it also has a YDS
    # rating.
    return routes_df.dropna(subset=["boulder_grade"])

def translate_boulder_grade(routes_df):
    routes_df["translated_grade"] = routes_df["boulder_grade"].str.removeprefix("V")
    routes_df["translated_grade"] = routes_df["translated_grade"].map(translate_boulder_sign)
    return routes_df

def translate_boulder_sign(grade_string):
    grade = re.search(r"^\d{1,2}|[eE]|[bB]", grade_string)
    if grade == None:
        return "-1"
    else: 
        grade = grade.group()
    sign = re.search(r"\+|\-$", grade_string)
    if sign != None:
        sign = sign.group()

    if grade.upper() == "B" or grade.upper() == "E":
        grade = 0
    grade = int(grade)

    if sign == "+":
        grade += 0.25
    elif sign == "-":
        grade -= 0.25

    return grade
    
