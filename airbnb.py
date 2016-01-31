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
    
#variance doesn't seem as strong here

#look at distribution across countries
age_gender_summary_stats.groupby("country_destination").population_in_thousands.sum().plot(kind = "bar")
#US is most populous country for sure, while Germany, France, FB, and Italy trail behind 

#what about proportional distribution across countries?
total_pop = age_gender_summary_stats.groupby("country_destination").population_in_thousands.sum().sum()
(age_gender_summary_stats.groupby("country_destination").population_in_thousands.sum() / total_pop * 100.0)
#US is at 44%

#seems like there's a mistake... this is showing the population of the countries, not the people in each segment.
# let's pull in country data

country_summary = pd.read_csv("data/countries.csv")
country_summary.head(10)
country_summary.info()


### USERS TABLE
#pull in users
users = pd.read_csv("data/train_users_2.csv")
users.info()
users.head()
users.dtypes

users.date_account_created.value_counts() #need to change date variables to date times

#going to make a dummy for Booked or Not Booked
users["booked"] = users.country_destination != "NDF"
users.booked.sum() *100.0 / len(users) #41.7% of people booked

#do some EDA
users.gender.value_counts()
users.groupby("gender").country_destination.count()

for gender in users.gender.unique():
    f1 = plt.figure()
    f1.add_subplot(users[users.gender == gender].country_destination.value_counts().plot(kind = "bar", title = gender))
    plt.show()

users[users.country_destination == "US"].gender.value_counts().plot(kind = "bar")
for country in users.country_destination.unique():
    f1 = plt.figure()
    f1.add_subplot(users[users.country_destination == country].gender.value_counts().plot(kind = "bar", title = country))
    plt.show()

#cool: significant variation of gender distribution across countries. Seems like gender will be a good predictor.

for gender in users.gender.unique():
    f1 = plt.figure()
    f1.add_subplot(users[users.gender == gender].booked.value_counts().plot(kind = "bar", title = gender))
    plt.show()

#investigate age
users.age.value_counts(dropna = False).sort_index() #looks like there are some weird anomalies with age
users[(users.age <= 110) | pd.isnull(users.age)]

for country in users.country_destination.unique():
    f1 = plt.figure()
    f1.add_subplot(users[users.country_destination == country].age.value_counts().sort_index().plot(kind = "bar", title = country))
    plt.show()
    
#distributions don't seem that different

users[pd.isnull(users.age)].country_destination.value_counts().plot(kind = "bar")
#most people without an age did not book. makes sense.

#Explore language
users.language.value_counts(dropna = False)

for lang in users.language.unique():
    f1 = plt.figure()
    f1.add_subplot(users[users.language == lang].country_destination.value_counts().plot(kind = "bar", title = lang))
    plt.show()

#looks like there are some interesting patterns with language!
#some other quick eda
users.info()
users.signup_flow.value_counts()
users.first_browser.value_counts()
users.first_device_type.value_counts()
users.signup_app.value_counts()
users.affiliate_channel.value_counts()
users.affiliate_provider.value_counts()
users.signup_method.value_counts()

#okay, let's convert stuff to date/time
users.info()
users.date_account_created = pd.to_datetime(users.date_account_created)
users.date_first_booking = pd.to_datetime(users.date_first_booking)
users.timestamp_first_active = pd.to_datetime(users.timestamp_first_active.apply(lambda x: int(str(x)[0:8])), format = "%Y%m%d")

(users.date_account_created - users.timestamp_first_active).value_counts().sort_index() #most people sign up same day... timestamp first active might not be very useful

(users.date_first_booking - users.date_account_created).value_counts()[0:30].sort_index() #most people book soon after making an account

#add month account was created, since I think that might be important...
users["month_account_created"] = users.date_account_created.apply(lambda x: x.month)
users.month_account_created.value_counts().sort_index().plot(kind = "bar") #more people create accounts just before summer... not surprising

users["weekday_account_created"] = users.date_account_created.apply(lambda x: x.weekday())
users.weekday_account_created.value_counts().sort_index().plot(kind = "bar")