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
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

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

#add month account was created, since I think that might be important due to holidays, weather, etc...
users["month_account_created"] = users.date_account_created.apply(lambda x: x.month)
users.month_account_created.value_counts().sort_index().plot(kind = "bar") #more people create accounts just before summer... not surprising

users["weekday_account_created"] = users.date_account_created.apply(lambda x: x.weekday())
users.weekday_account_created.value_counts().sort_index().plot(kind = "bar")


users.info()

sessions = pd.read_csv("data/sessions.csv")

sessions.info()
sessions.head(40)
sessions.action.value_counts()
sessions.action_type.value_counts()
sessions.action_detail.value_counts()
sessions.device_type.value_counts()

sessions[sessions.action == "show"].action_type.value_counts()
sessions[sessions.action == "show"].action_detail.value_counts() #looks like it's related to opening certain views on the website

sessions[sessions.action == "index"].action_type.value_counts()
sessions[sessions.action == "index"].action_detail.value_counts() #unclear

sessions[sessions.action == "search_results"].action_type.value_counts()
sessions[sessions.action == "search_results"].action_detail.value_counts() #view search results

sessions[sessions.action == "personalize"].action_type.value_counts()
sessions[sessions.action == "personalize"].action_detail.value_counts() #wish list!

sessions[sessions.action == "search"].action_type.value_counts()
sessions[sessions.action == "search"].action_detail.value_counts() # just view search results

sessions[sessions.action == "ajax_refresh_subtotal"].action_detail.value_counts() #change trip characteristics

sessions[sessions.action == "similar_listings"].action_detail.value_counts() #similar listings detail

sessions[sessions.action == "social_connections"].action_detail.value_counts() #user social connections... intersting

sessions[sessions.action == "reviews"].action_detail.value_counts() #someone looking at listing/user reviews?

sessions[sessions.action_detail == "-unknown-"].action.value_counts() #someone looking at listing/user reviews?

sessions.action.value_counts()[sessions.action.value_counts() > 1000]
sessions.action.value_counts()[sessions.action.value_counts() > 1000].to_csv("test.csv")

'''
Okay, cool... here's what I'm going to do.
1) Count up all actions for a user, store as column
2) For each action > 1000, I'm going to loop through the actions,
and count up how many each user has
'''

#count up sessions per user
actions_per_user = pd.DataFrame(sessions.groupby("user_id").action.count())
actions_per_user

users.info()

users = users.merge(actions_per_user, left_on = "id", right_index = True, how = "left")
users.fillna(value = dict(action=0), inplace = True)
#count up each action > 1000 for each user


user_ids_to_merge_with_action_counts = pd.DataFrame({"user_id": sessions.user_id.unique()})
for action in sessions.action.value_counts()[sessions.action.value_counts() > 1000].index:
    current_actions_only = sessions[sessions.action == action]
    
    action_count = pd.DataFrame({action: current_actions_only.groupby("user_id").action.count()})
    
    user_ids_to_merge_with_action_counts = user_ids_to_merge_with_action_counts.merge(action_count, left_on = "user_id", right_index = True, how = "left")


user_ids_to_merge_with_action_counts.fillna(0, inplace = True)

users = users.merge(user_ids_to_merge_with_action_counts, left_on = "id", right_on = "user_id", how = "left")
fill_na_after_merge = pd.DataFrame([0] * 148, index = user_ids_to_merge_with_action_counts.columns).to_dict()[0]

users.fillna(value = fill_na_after_merge, inplace = True)
fill_na_after_merge

avg_age_by_gender = users.groupby("gender").age.mean()
avg_age_by_gender
for gender in users.gender.unique():
    users.loc[users[(users.gender == gender) & (pd.isnull(users.age))].index, "age"] = avg_age_by_gender[gender]

users.age.value_counts(dropna = False)

pd.DataFrame(users.columns).to_csv("user_columns.csv")

'''
need to get dummies for:
gender
signup_method
signup_flow
language
affiliate_channel
affiliate_provider
first_affiliate_tracked
signup_app
first_device_type
first_browser
'''
gender_dummies = pd.get_dummies(users.gender, prefix = "gender")
users = users.merge(gender_dummies, left_index = True, right_index = True)

users.signup_method
signup_method_dummies = pd.get_dummies(users.signup_method, prefix = "signup_method")
users = users.merge(signup_method_dummies, left_index = True, right_index = True)

users.signup_flow.value_counts()
signup_flow_dummies = pd.get_dummies(users.signup_flow, prefix = "signup_flow")
users = users.merge(signup_flow_dummies, left_index = True, right_index = True)

users.language.value_counts()
language_dummies = pd.get_dummies(users.language, prefix = "language")
users = users.merge(language_dummies, left_index = True, right_index = True)

users.affiliate_channel.value_counts()
affiliate_channel_dummies = pd.get_dummies(users.affiliate_channel, prefix = "affiliate_channel")
users = users.merge(affiliate_channel_dummies, left_index = True, right_index = True)

users.affiliate_provider.value_counts()
affiliate_provider_dummies = pd.get_dummies(users.affiliate_provider, prefix = "affiliate_provider")
users = users.merge(affiliate_provider_dummies, left_index = True, right_index = True)

users.first_affiliate_tracked.value_counts()
first_affiliate_tracked_dummies = pd.get_dummies(users.first_affiliate_tracked, prefix = "first_affiliate_tracked")
users = users.merge(first_affiliate_tracked_dummies, left_index = True, right_index = True)

users.signup_app.value_counts()
signup_app_dummies = pd.get_dummies(users.signup_app, prefix = "signup_app")
users = users.merge(signup_app_dummies, left_index = True, right_index = True)

users.first_device_type.value_counts()
first_device_type_dummies = pd.get_dummies(users.first_device_type, prefix = "first_device_type")
users = users.merge(first_device_type_dummies, left_index = True, right_index = True)

users.first_browser.value_counts()
first_browser_dummies = pd.get_dummies(users.first_browser, prefix = "first_browser")
users = users.merge(first_browser_dummies, left_index = True, right_index = True)

pd.DataFrame(users.columns).to_csv("user_columns.csv")


#let's build a model!


rfc = RandomForestClassifier(random_state = 0, n_estimators = 300, oob_score = True)
X_cols = pd.read_csv("x_cols_m1.csv", header = None)[0]
X_cols
X = users[X_cols]
y = users.booked

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
rfc.fit(X_train, y_train)

rfc.oob_score_

rfc.score(X_test, y_test)

y_test.sum() * 1.0 / len(y_test)