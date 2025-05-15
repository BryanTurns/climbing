import pandas as pd
import re


def general_clean(routes_df):
    routes_df["avg_rating"] = pd.to_numeric(routes_df["avg_rating"])
    routes_df["rating_count"] = pd.to_numeric(routes_df["rating_count"])

    routes_df = routes_df[routes_df["rating_count"] > 5]

    return routes_df

def ropes_only(routes_df):
    routes_df = routes_df.dropna(subset=["grade"])
    routes_df = routes_df[routes_df["grade"].str.contains(r"^[^V].*", regex=True)]
    return routes_df

def bouldering_only(routes_df):
    routes_df = routes_df.dropna(subset=["grade"])
    routes_df = routes_df[routes_df["grade"].str.startswith("V")]
    return routes_df

def translate_boulder_grade(routes_df):
    routes_df["translated_grade"] = routes_df["grade"].str.removeprefix("V")
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
    
