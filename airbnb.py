# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:53:34 2016

@author: asjedh
"""
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

os.chdir("/Users/asjedh/Desktop/kaggle_airbnb")

age_gender_summary_stats = pd.read_csv("data/age_gender_bkts.csv")

age_gender_summary_stats.info()
age_gender_summary_stats.head(50)
#year is just 2015. 
age_gender_summary_stats.age_bucket.value_counts()
age_gender_summary_stats.country_destination.value_counts()
age_gender_summary_stats.population_in_thousands.describe() #population for age-country-gender
age_gender_summary_stats.population_in_thousands.hist(bins = 50, by = age_gender_summary_stats.gender)
age_gender_summary_stats.population_in_thousands.hist(bins = 50, by = age_gender_summary_stats.country_destination)

#all countries by gender
age_gender_summary_stats.groupby("gender").population_in_thousands.sum().plot(kind="bar")

for country in age_gender_summary_stats.country_destination.unique():
    f1 = plt.figure()
    f1.add_subplot(age_gender_summary_stats[age_gender_summary_stats.country_destination == country].groupby("gender").population_in_thousands.sum().plot(kind="bar", title = country))
    plt.show()

#relatively similar proportions! seems like Males like AU and NL a bit more than average, while females like FR

#all countries by age group

order_of_ages = ["0-4", "5-9", "10-14", "15-19", "20-24",\
"25-29", "30-34", "35-39", "40-44", "45-49", "50-54",\
"55-59", "60-64", "65-69", "70-74", "75-79", "80-84",\
"85-89", "90-94", "95-99", "100+"]

age_gender_summary_stats.groupby("age_bucket").population_in_thousands.sum()[order_of_ages].plot(kind="bar")

for country in age_gender_summary_stats.country_destination.unique():
    f1 = plt.figure()
    f1.add_subplot(age_gender_summary_stats[age_gender_summary_stats.country_destination == country].groupby("age_bucket").population_in_thousands.sum()[order_of_ages].plot(kind="bar", title = country))
    plt.show()

#different distributions of ages!

# group by age and gender

order_of_gender_ages_female = [("female", age) for age in order_of_ages]
order_of_gender_ages_male = [("male", age) for age in order_of_ages]
order_of_gender_ages = order_of_gender_ages_female + order_of_gender_ages_male


age_gender_summary_stats.groupby(["gender", "age_bucket"]).population_in_thousands.sum()[order_of_gender_ages].plot(kind="bar")

for country in age_gender_summary_stats.country_destination.unique():
    f1 = plt.figure()
    f1.add_subplot(age_gender_summary_stats[age_gender_summary_stats.country_destination == country].groupby(["gender", "age_bucket"]).population_in_thousands.sum()[order_of_gender_ages].plot(kind="bar", title = country))
    plt.show()
    


