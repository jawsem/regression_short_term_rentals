## Regression Analysis of Short Term Rentals

**This is a repository where Kaggle Airbnb data was used for a Capstone project for my WGU grad program.  Jupyter Notebooks were used to analyze and clean data and the data was fed through a few different modelling techniques.  The performance of each technqiue was compared and a capstone paper was written on it.**

In order acquire all the data needed to run this project follow the steps below.

1. download airbnb csv from kaggle
https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data/downloads/new-york-city-airbnb-open-data.zip/3
You can save this file in the datasets folder AB_NYC_2019.csv

2. get new york places from google public datasets

https://console.cloud.google.com/bigquery (get trees)
SELECT * FROM `bigquery-public-data.new_york.tree_census_2015`;
newyorktrees.csv

SELECT * FROM `bigquery-public-data.geo_us_census_places.places_new_york`;
get new york census tracts from google public datasets
newyorkcensustracts.csv

3. get subway stations
SELECT * FROM `bigquery-public-data.new_york_subway.station_entrances`;
newyorksubwayent.csv
4. Download data from tractfinder

https://factfinder.census.gov/faces/nav/jsf/pages/download_center.xhtml#none

2017 ACS 5 year estimates

see screenshot factfinder.jpg

exctract out aff_download.zip

The overall project is saved in Jawsem Capstone_Regression Analysis of Short Term Rentals.pdf.  The majority of the code 
is written in the ipython notebook.  Some additional ideas for the project are:

1. Acquire the data automatically via API.
2. Extract more census data that could be more indicative of the price of a AirBnb.
3. Add interactions onto the linear regression models.