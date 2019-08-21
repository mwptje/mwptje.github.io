---
title: "Battle of the Neighbourhoods"
date: 2019-02-23
tags: [data science, data wrangling, python, geo-location]
header:
  image: "/images/capstone/rohan-makhecha-unsplash.jpg"
excerpt: "Data Wrangling, Data Science, Python, Geo-location"
mathjax: "true"
---
Image by Rohan Makhecha on Unsplash

## Capstone Project - Battle of the Neighbourhoods
Using unsupervised machine learning to categorize neighbourhoods to 
provide additonal information as to where to locate certain businesses 
in the city of Toronto.

Using: Python, Jupyter Notebook, statistical and spatial data

## Contents
* [Introduction: Business Problem](#intro)
* [Data](#data)
  * [Statistical Data on Neighbourhoods](#stats)
  * [Foursquare API data - Venue Details](#foursquare)
* [Data Exploration / Methodology](#methodology)
* [Analysis](#analysis)

## Introduction / Business Problem <a name="intro"></a>

Finding the right small business location is one of the primary steps in preparing to set up a new business. It is not always an easy task. This project aims to help current and future business owners in the process of selecting business locations. By using data from a location based social network services like Foursquare as well as neighbourhood area statistics it should be possible to recommend possible business locations.

As the types of small businesses are manifold, this project will restrict the definition to those businesses that fall under the categories of shops, service venues, restaurants, cafes and bars. These types of businesses depend on foot traffic and/or easy access by car or public transport and good visibility.

### There are several of factors that can influence choosing a location:

* Location of similar businesses
 * Businesses are usually located where they are for a good reason,
 * Customers already in the area are more likely to be looking for a similar business
* Consumer statistics for similar business
 * Average number of customer visits. 
 * Popularity of a business
* Distance between consumers and business
 * The further the consumer is located from the business the less likely he or she is to visit.
 * Consumer location doesn't necesarily mean domestic location but could also mean job location.
* Location close to transportation hubs, parking facilities, entertainment centres like theatres, cinemas or public parks
 * Locations where there is a large amount of foot traffic: concentration of possible customers
* Population density of the surrounding area
 * More people close by: more possible customers
 * There are statistics available on population by neighbourhood or postal code area. 
* Average Income 
 * Higher average income: possible customers with more money to spend
 * There are statistics available on average income by neighbourhood. 

  **This project will attempt to combine the above factors to build a clustering and/or recommendation model
  for the best areas for locating certain businesses. The recommendation(s) given by the model should help
  the (future) business owner to make a more informed decision**

**Note**: only further analysis in the next stage after gathering the data will prove which machine learning method is better suited to use

Staring off with importing the necessary Python libraries and setting pandas display options


```python
# import the necessary libraries
import os
import numpy as np # library to handle data in a vectorized manner
import pandas as pd # library for data analsysis
import geopandas as gpd # libary for geo-spatial data processing and analysis
# import the Point object
from shapely.geometry import Point
import json # library to handle JSON files
import requests # library to handle requests
import pickle # library to save serialized
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
# graphical libraries
import matplotlib.pyplot as plt
import seaborn as sns
import folium
# no warnings
import warnings
warnings.filterwarnings('ignore')
# we need some modules from scikit-learn
from sklearn import preprocessing
# import k-means for the clustering stage
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# show plots inline
%matplotlib inline
```

## Data Section <a name="data"/>

## Statistics Data on Neighbourhoods <a name="stats"/>

I have chosen to look at the neighbourhoods in the former city of Toronto for this study. This is based on the fact that the city has a substantially large population with readily available statistics.

### 1. Neighbourhoods with central and boundary geo-coordinates with the following columns:

 * **CDN_Number**: Area code for the neighbourhood, 3 digits
 * **Neighbourhood**: Name of the neighbourhood
 * **geometry**: collection of geo-coordinates designating the boundary of the neigbourhood
 * **Latitude**: the latitudinal coordinate of the center of the area (centroid)
 * **Longitude**: the longitudinal coordinate of the center of the area (centroid)
 
 **Neighbourhood**: according to the website of the city of Toronto, the definition of a neighbourhood
 is an area that respects existing boundaries such as service boundaries of community agencies, 
 natural boundaries (rivers), and man-made boundaries (streets, highways, etc.)
 They are small enough for service organizations to combine them to fit within their service area.
 They represent municipal planning areas as well as areas for public service like public health.
 A neighbourhood has a population roughly between 7,000 and 12,00 people.

### Spatial data on the neighbourhoods of Toronto:
 * Using geopandas read_file method to convert a Shapefile into a dataframe format
 * Rename columns to be consistent when joining dataframes later on
 * Use geopandas centroid method to determine the geo-coordinaties of the center of a neighbourhood
 * Display the first few rows and the number of rows and columns of the dataframe

```python
# convert the neighbourhood's boundaries shapefile to a geopandas dataframe
df_toronto_nbh_geo = gpd.read_file('./data/NEIGHBORHOODS_WGS84.shp')
# rename the columns
df_toronto_nbh_geo.rename(columns={'AREA_S_CD':'CDN_Number', 'AREA_NAME':'Neighbourhood'}, inplace=True)
# remove the brackets in the neighbourhod name column
fix_neighbourhood = lambda x: x.split('(')[0]
df_toronto_nbh_geo['Neighbourhood'] = df_toronto_nbh_geo['Neighbourhood'].apply(fix_neighbourhood)
# calculate the centers of each area 
df_toronto_nbh_geo['Latitude'] = df_toronto_nbh_geo['geometry'].centroid.y
Capstone df_toronto_nbh_geo['Longitude'] = df_toronto_nbh_geo['geometry'].centroid.x
# display the dimensions and first five rows
print('Dimensions: ', df_toronto_nbh_geo.shape)
df_toronto_nbh_geo.head()
```

    Dimensions:  (140, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDN_Number</th>
      <th>Neighbourhood</th>
      <th>geometry</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>097</td>
      <td>Yonge-St.Clair</td>
      <td>POLYGON ((-79.39119482700001 43.681081124, -79...</td>
      <td>43.687859</td>
      <td>-79.397871</td>
    </tr>
    <tr>
      <th>1</th>
      <td>027</td>
      <td>York University Heights</td>
      <td>POLYGON ((-79.505287916 43.759873494, -79.5048...</td>
      <td>43.765738</td>
      <td>-79.488883</td>
    </tr>
    <tr>
      <th>2</th>
      <td>038</td>
      <td>Lansing-Westgate</td>
      <td>POLYGON ((-79.439984311 43.761557655, -79.4400...</td>
      <td>43.754272</td>
      <td>-79.424747</td>
    </tr>
    <tr>
      <th>3</th>
      <td>031</td>
      <td>Yorkdale-Glen Park</td>
      <td>POLYGON ((-79.439687326 43.705609818, -79.4401...</td>
      <td>43.714672</td>
      <td>-79.457108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>016</td>
      <td>Stonegate-Queensway</td>
      <td>POLYGON ((-79.49262119700001 43.64743635, -79....</td>
      <td>43.635518</td>
      <td>-79.501128</td>
    </tr>
  </tbody>
</table>
</div>

<br>

**Note**:In the case of the neighbourhoods geospatial data no data cleansing is necessary, other than removing
removing the CDN number from the description. I have renamed the columns to be consistent.
The centeral geo-coordinates for each neighbourhood have also been calculated using a geopandas geometry attribute called centroid.

### 2. Wikipedia table containing neighbourhoods by former city / borough

  This table is used to filter the neighbourhoods by the former city area of Toronto: 
  https://en.wikipedia.org/wiki/List_of_city-designated_neighbourhoods_in_Toronto
  * **CDN_Number**: Area code for the neighbourhood, 3 digits
  * **City-designated-area**: Name of the neighbourhood
  * **Borough**: Former city or borough


```python
# read the table in the Wikipedia page
df_toronto_nbh_bor = pd.read_html('https://en.wikipedia.org/wiki/List_of_city-designated_neighbourhoods_in_Toronto')[0]
# remove columns not needed and rename the remaining
df_toronto_nbh_bor.drop(columns=['Map','Neighbourhoods covered'],inplace=True)
df_toronto_nbh_bor.rename(columns={'CDN number':'CDN_Number','Former city/borough':'Borough'}, inplace=True)
# format the CDN number column so that it matches that of the previous dataframe
zero_fill = lambda x: "{:03d}".format(x)
df_toronto_nbh_bor['CDN_Number'] = df_toronto_nbh_bor['CDN_Number'].apply(zero_fill)
# display the dimensions and first five rows
print('Dimensions: ', df_toronto_nbh_bor.shape)
df_toronto_nbh_bor.head()
```

    Dimensions:  (140, 3)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDN_Number</th>
      <th>City-designated area</th>
      <th>Borough</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>129</td>
      <td>Agincourt North</td>
      <td>Scarborough</td>
    </tr>
    <tr>
      <th>1</th>
      <td>128</td>
      <td>Agincourt South-Malvern West</td>
      <td>Scarborough</td>
    </tr>
    <tr>
      <th>2</th>
      <td>020</td>
      <td>Alderwood</td>
      <td>Etobicoke</td>
    </tr>
    <tr>
      <th>3</th>
      <td>095</td>
      <td>Annex</td>
      <td>Old City of Toronto</td>
    </tr>
    <tr>
      <th>4</th>
      <td>042</td>
      <td>Banbury-Don Mills</td>
      <td>North York</td>
    </tr>
  </tbody>
</table>
</div>



  **Note**: In the case of the wikipedia list of neighbourhoods in Toronto, there are now missing values.
  To be consistant, I have reformated the CDN number to a zero-fill 3 digit number.
  Just to make sure I compared the CDN numbers and neighbourhood names to the neighbourhood
  geospatial file and there were no differences. The number of rows (read neighbourhoods is the same)

### 3. Toronto Population Statistics by Neighbourhood

  Neighbourhood population , area and household income from 2014. 
  
  This can be retrieved from the city of Toronto neighbourhood wellbeing app
  https://www.toronto.ca/city-government/data-research-maps/neighbourhoods-communities/wellbeing-toronto/
  
  The file contains the following columns:
  * **Neighbourhood**: Name of the neighbourhood
  * **CDN_Number**: Three digit neighbourhood code
  * **TotalPopulation**: Total population for the neighbourhood based on 2014 data
  * **TotalArea**: Area of the neighbourhood in square kilometers
  * **After_TaxHouseholdIncome**: Average household income after tax in Canadian dollars
  * **PopulationDensity**: Density of the population by square kilometers
  
  This excel file will be loaded into a pandas dataframe 

### Example data:


```python
# read the 2014 statistics excel file
df_toronto_nbh_sta = pd.read_excel('./data/wellbeing_toronto_2014.xlsx')
# remove unwanted columns
df_toronto_nbh_sta.drop(columns=['Combined Indicators','Average Family Income'],inplace=True)
# rename the neighbourhood id column to CDN_Number to match other dataframe
rename_columns = {'Neighbourhood Id':'CDN_Number',
                  'Total Population':'TotalPopulation',
                  'Total Area':'TotalArea',
                  'After-Tax Household Income':'AfterTaxHouseholdIncome'}
df_toronto_nbh_sta.rename(columns=rename_columns,inplace=True)
# reformat the CDN_Number column to match the other similar dataframe columns
zero_fill = lambda x: "{:03d}".format(x)
df_toronto_nbh_sta['CDN_Number'] = df_toronto_nbh_sta['CDN_Number'].apply(zero_fill)
# add column with population density
df_toronto_nbh_sta['PopulationDensity'] = round(df_toronto_nbh_sta['TotalPopulation']/df_toronto_nbh_sta['TotalArea'],0)
df_toronto_nbh_sta.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>CDN_Number</th>
      <th>TotalPopulation</th>
      <th>TotalArea</th>
      <th>AfterTaxHouseholdIncome</th>
      <th>PopulationDensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>West Humber-Clairville</td>
      <td>001</td>
      <td>33312</td>
      <td>30.09</td>
      <td>59703</td>
      <td>1107.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mount Olive-Silverstone-Jamestown</td>
      <td>002</td>
      <td>32954</td>
      <td>4.60</td>
      <td>46986</td>
      <td>7164.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Thistletown-Beaumond Heights</td>
      <td>003</td>
      <td>10360</td>
      <td>3.40</td>
      <td>57522</td>
      <td>3047.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rexdale-Kipling</td>
      <td>004</td>
      <td>10529</td>
      <td>2.50</td>
      <td>51194</td>
      <td>4212.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Elms-Old Rexdale</td>
      <td>005</td>
      <td>9456</td>
      <td>2.90</td>
      <td>49425</td>
      <td>3261.0</td>
    </tr>
  </tbody>
</table>
</div>



  **Note**: There were no empty values in this table and the number of rows compared with the previous
  neighbourhood files. To be consistent, I have reformatted the CDN number to a zero-fill 3 digit number.
  Just to make sure I compared the CDN numbers and neighbourhood names to the neighbourhood 
  geospatial file and there were no differences. A population density column was calculated by dividing
  the total population by the total area of the neighbourhood.

### 4. Combined table with neighbourhood as key

  The three above mentioned tables will be loaded and joined based on FSA code to form a dataframe containing 
  the following columns:
  
 * **CDN_Number**: Three digits designating a neighbourhood (data 1.)
 * **Neighbourhood**: Name of the neighbourhood (data 1.)
 * **Latitude**: the latitudinal coordinate of the center of the area (data 1.)
 * **Longitude**: the longitudinal coordinate of the center of the area (data 1.)
 * **geometry**: a list of latitude - longitude coordinates forming the boundaries of the neighbourhood (data 1.)
 * **TotalPopulation**: the total population of the neighbourhood (data 3.)
 * **TotalArea**: the total area in square kilometers (data 3.)
 * **AfterTaxHouseholdIncome**: average household income after tax for the neighbourhod (data 3.)
 * **PopulationDensity**: the population density of the area in persons by square km (TotalPopulation/TotalArea)

  This dataframe named **df_toronto_ven** will form the features for a neighbourhod and used for the machine learning algorithm

### Example data:

  Only the neighbourhoods in the former city of Toronto have been retained
  After removing several (duplicate) columns, the following columns are available as shown below:


```python
# Now join the three dataframes
df_toronto_nbh_tmp = pd.merge(left=df_toronto_nbh_geo,right=df_toronto_nbh_bor,on='CDN_Number')
df_toronto_nbh_tmp = df_toronto_nbh_tmp[df_toronto_nbh_tmp['Borough'] == 'Old City of Toronto']
df_toronto_nbh_tmp.drop(columns=['City-designated area','Borough'],inplace=True)
#df_toronto_nbh.rename(columns={'NeighbourhoodGeo':'Neighbourhood'},inplace=True)
df_toronto_nbh = pd.merge(left=df_toronto_nbh_tmp,right=df_toronto_nbh_sta,on='CDN_Number')
df_toronto_nbh.drop(columns=['Neighbourhood_y'],inplace=True)
df_toronto_nbh.rename(columns={'Neighbourhood_x':'Neighbourhood'},inplace=True)
df_toronto_nbh.sort_values('Neighbourhood',inplace=True)
df_toronto_nbh.reset_index(drop=True,inplace=True)
print('Dimensions: ',df_toronto_nbh.shape)
df_toronto_nbh.head()
```

    Dimensions:  (44, 9)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDN_Number</th>
      <th>Neighbourhood</th>
      <th>geometry</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>TotalPopulation</th>
      <th>TotalArea</th>
      <th>AfterTaxHouseholdIncome</th>
      <th>PopulationDensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>095</td>
      <td>Annex</td>
      <td>POLYGON ((-79.39414141500001 43.668720261, -79...</td>
      <td>43.671585</td>
      <td>-79.404000</td>
      <td>30526</td>
      <td>2.8</td>
      <td>49912</td>
      <td>10902.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>076</td>
      <td>Bay Street Corridor</td>
      <td>POLYGON ((-79.38751633 43.650672917, -79.38662...</td>
      <td>43.657512</td>
      <td>-79.385722</td>
      <td>25797</td>
      <td>1.8</td>
      <td>44614</td>
      <td>14332.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>069</td>
      <td>Blake-Jones</td>
      <td>POLYGON ((-79.34082169200001 43.669213123, -79...</td>
      <td>43.676173</td>
      <td>-79.337394</td>
      <td>7727</td>
      <td>0.9</td>
      <td>51381</td>
      <td>8586.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>071</td>
      <td>Cabbagetown-South St.James Town</td>
      <td>POLYGON ((-79.376716938 43.662418858, -79.3772...</td>
      <td>43.667648</td>
      <td>-79.366107</td>
      <td>11669</td>
      <td>1.4</td>
      <td>50873</td>
      <td>8335.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>096</td>
      <td>Casa Loma</td>
      <td>POLYGON ((-79.414693177 43.673910413, -79.4148...</td>
      <td>43.681852</td>
      <td>-79.408007</td>
      <td>10968</td>
      <td>1.9</td>
      <td>65574</td>
      <td>5773.0</td>
    </tr>
  </tbody>
</table>
</div>



### Display the neighbourhoods on a map of Toronto by population density

  Each neighbourhood is shown with a boundary and a color varying from yellow to red, depending on the population density by square kilometer. This is a preliminary exploration into the data we have gathered.


```python
# create map of Toronto Neighbourhoods (FSAs) using retrived latitude and longitude values
map_toronto = folium.Map(location=[43.673963, -79.387207], zoom_start=12);
toronto_geojson = "./data/toronto_neighbourhoods.json"
map_toronto.choropleth(geo_data=toronto_geojson,
    data = df_toronto_nbh,
    popup=df_toronto_nbh['Neighbourhood'],
    columns=['Neighbourhood','PopulationDensity'],
    key_on='feature.properties.Neighbourhood',
    fill_color='YlOrRd',
    fill_opacity=0.5, 
    line_opacity=0.2,
    legend_name='Population Density by Neighbourhood')   
# add markers to map
for lat, lng, cdn_number, neighborhood in zip(df_toronto_nbh['Latitude'], df_toronto_nbh['Longitude'], df_toronto_nbh['CDN_Number'], df_toronto_nbh['Neighbourhood']):
    label = '{} - {}'.format(neighborhood, cdn_number)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=2,
        popup=label,
        color='red',
        fill=True,
        #fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_toronto)  
map_toronto.save('toronto_map.html')
```

<img src='/images/capstone/toronto_map.jpeg'/>

## Foursquare API data - Venue Details <a name="foursquare" />

The Foursquare API will be used to collect venue data by FSA area. 
This data can then be combined with the FSA statistical data 
to be used by the chosen machine learning algorithm to provide insight in business location


```python
# Set up Foursqaure API credentials
CLIENT_ID = '<client id here>' # your Foursquare ID
CLIENT_SECRET = '<client credentials here>' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
```

### 5. Foursquare Venue Categories:
 
  Each venue on Foursqaure has been assigned to a category.
  This is is the lowest level category that is used by Foursqaure.
  
  Foursqaure usually has two levels of categories, the top level like Food, Arts & Entertainment etc.
  Under each category there are several sub-categories.
  For example Food has a long list of sub-categories including different restaurant types, cafes etc.
  
  There is a special entry point in the Foursqaure API to retrieve all categories and sub-categories.
  This data will be stored in a table with the following fields:
  
  * **Category**: top level Foursquare venue catagory
  * **Subcategory**: lower level venue category
  
  The top level category will be used to categorize venues on a top level as well


```python
# build the request to retrieve the Foursquare venue catagories
url = 'https://api.foursquare.com/v2/venues/categories?&client_id={}&client_secret={}&v={}'.format(
   CLIENT_ID, 
   CLIENT_SECRET, 
   VERSION
) 
# initialize variables
dict_cats = {}
list_cats = []
list_subcats = []
# check if the categories csv file already exists, if so then use it
# instead of calling the API
if os.path.exists('data/foursquare_categories.csv'):
    df_cats=pd.read_csv('data/foursquare_categories.csv',index_col=0)
else:
    # request the data from the API
    results = requests.get(url).json()
    # normalize the Json to a dataframe
    df_cats = json_normalize(results['response']['categories'])
    # get each category and sub-category from the categories column
    for idx,row in df_cats.iterrows():
        cats = row['categories']
        for v in cats:
            list_cats.append(row['name'])
            list_subcats.append(v['name'])
    dict_cats['Category'] = list_cats
    dict_cats['Subcategory'] = list_subcats
    # rebuild the dataframe from a dictionary
    df_cats = pd.DataFrame.from_dict(dict_cats)
    # save to csv for later use
    df_cats.to_csv('data/foursquare_categories.csv')
    
df_cats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Category</th>
      <th>Subcategory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Arts &amp; Entertainment</td>
      <td>Amphitheater</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arts &amp; Entertainment</td>
      <td>Aquarium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arts &amp; Entertainment</td>
      <td>Arcade</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arts &amp; Entertainment</td>
      <td>Art Gallery</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Arts &amp; Entertainment</td>
      <td>Bowling Alley</td>
    </tr>
  </tbody>
</table>
</div>



### 6. Foursquare Venues by Neighbourhood

  Use the Foursquare Venue Explore API endpoint to gather basic data on venues
  with a certain radius based on the central coordinates for the area. 
  
  The data retrieed in JSON format will be stored in a dataframe with the following columns:
  
  * **CDN_Number**: Three digit neighbourhood code
  * **Neighbourhood**: Name of the neighbourhood the venue is located in
  * **Name**: Name of the venue
  * **Latitude**: Latitude coordinate of the venue
  * **Longitude**: Longitude coordinate of the venue
  * **Subcategory**: Lower level category name for the venue
  * **Category**: Highest level category , this will be added later

  **Note**: the venue category will be added to the dataframe using the Foursquare's categories dataframe (5)
  **Note**: the venue CDN number and neighbourhood will be checked against the neighbourhoods boundaries

### Get the venues by neighbourhood using the Foursquare API explore endpoint


```python
def get_nearby_venues(cdns, neighbourhoods, latitudes, longitudes, radius=1000):
    
    venues_list=[]
    for cdn, neighbourhood, lat, lng in zip(cdns, neighbourhoods, latitudes, longitudes):
        print(cdn,'-',neighbourhood)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        venues = requests.get(url).json()["response"]['groups'][0]['items']
        # add a row for each venue
        for v in venues:
            vnam = v['venue']['name']             # venue name
            vlat = v['venue']['location']['lat']  # venue latitude
            vlng = v['venue']['location']['lng']  # venue longitude
            vcat = v['venue']['categories'][0]['name'] # venue subcategory
            venues_list.append([cdn,neighbourhood,vnam,vlat,vlng,vcat])            

    return(venues_list)
```

### Process retrieving venues by neighbourhood

  Loop through the neighbourhood dataframe to get the venues within a certain radius
  of the center coordinates of each neighbourhood. Due to the fact that using a radius might
  cause the API to get venues just outside of the current neighbourhood. All the venues found
  will be verified and if necessary set to the correct neighbourhood


```python
LIMIT = 200 # limit of number of venues returned by Foursquare API
radius = 1000 # define radius in meters
# call the API explore endpoint for each neighbourhood
venues_list = get_nearby_venues(cdns=df_toronto_nbh['CDN_Number'],
                                neighbourhoods=df_toronto_nbh['Neighbourhood'],
                                latitudes=df_toronto_nbh['Latitude'],
                                longitudes=df_toronto_nbh['Longitude']
                               )
```

    095 - Annex 
    076 - Bay Street Corridor 
    069 - Blake-Jones 
    071 - Cabbagetown-South St.James Town 
    096 - Casa Loma 
    075 - Church-Yonge Corridor 
    092 - Corso Italia-Davenport 
    066 - Danforth 
    093 - Dovercourt-Wallace Emerson-Junction 
    083 - Dufferin Grove 
    062 - East End-Danforth 
    102 - Forest Hill North 
    101 - Forest Hill South 
    065 - Greenwood-Coxwell 
    088 - High Park North 
    087 - High Park-Swansea 
    090 - Junction Area 
    078 - Kensington-Chinatown 
    105 - Lawrence Park North 
    103 - Lawrence Park South 
    084 - Little Portugal 
    073 - Moss Park 
    099 - Mount Pleasant East 
    104 - Mount Pleasant West 
    082 - Niagara 
    068 - North Riverdale 
    074 - North St.James Town 
    080 - Palmerston-Little Italy 
    067 - Playter Estates-Danforth 
    072 - Regent Park 
    086 - Roncesvalles 
    098 - Rosedale-Moore Park 
    089 - Runnymede-Bloor West Village 
    085 - South Parkdale 
    070 - South Riverdale 
    063 - The Beaches 
    081 - Trinity-Bellwoods 
    079 - University 
    077 - Waterfront Communities-The Island 
    091 - Weston-Pellam Park 
    064 - Woodbine Corridor 
    094 - Wychwood 
    100 - Yonge-Eglinton 
    097 - Yonge-St.Clair 


### Build the venues dataframe from the venues list and rename the columns


```python
# build the dataframe from the venues list
df_toronto_ven = pd.DataFrame.from_records(venues_list)
# rename the columns
df_toronto_ven.columns = ['CDN_Number',
              'Neighbourhood', 
              'Venue', 
              'Latitude', 
              'Longitude', 
              'SubCategory']
# display the first 5 rows
print('Dimensions: ', df_toronto_ven.shape)
df_toronto_ven.head()
```

    Dimensions:  (3411, 6)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDN_Number</th>
      <th>Neighbourhood</th>
      <th>Venue</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>SubCategory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>095</td>
      <td>Annex</td>
      <td>Rose &amp; Sons</td>
      <td>43.675668</td>
      <td>-79.403617</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>095</td>
      <td>Annex</td>
      <td>Ezra's Pound</td>
      <td>43.675153</td>
      <td>-79.405858</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>2</th>
      <td>095</td>
      <td>Annex</td>
      <td>Roti Cuisine of India</td>
      <td>43.674618</td>
      <td>-79.408249</td>
      <td>Indian Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>095</td>
      <td>Annex</td>
      <td>Fresh on Bloor</td>
      <td>43.666755</td>
      <td>-79.403491</td>
      <td>Vegetarian / Vegan Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>095</td>
      <td>Annex</td>
      <td>Playa Cabana</td>
      <td>43.676112</td>
      <td>-79.401279</td>
      <td>Mexican Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



### Add the category column based on a dictionary lookup using the Foursquare categories dataframe


```python
dict_cats = dict(zip(df_cats['Subcategory'],df_cats['Category']))
df_toronto_ven['Category'] = df_toronto_ven['SubCategory'].map(dict_cats)
df_toronto_ven.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDN_Number</th>
      <th>Neighbourhood</th>
      <th>Venue</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>SubCategory</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>095</td>
      <td>Annex</td>
      <td>Rose &amp; Sons</td>
      <td>43.675668</td>
      <td>-79.403617</td>
      <td>American Restaurant</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>1</th>
      <td>095</td>
      <td>Annex</td>
      <td>Ezra's Pound</td>
      <td>43.675153</td>
      <td>-79.405858</td>
      <td>Café</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>2</th>
      <td>095</td>
      <td>Annex</td>
      <td>Roti Cuisine of India</td>
      <td>43.674618</td>
      <td>-79.408249</td>
      <td>Indian Restaurant</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>3</th>
      <td>095</td>
      <td>Annex</td>
      <td>Fresh on Bloor</td>
      <td>43.666755</td>
      <td>-79.403491</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>4</th>
      <td>095</td>
      <td>Annex</td>
      <td>Playa Cabana</td>
      <td>43.676112</td>
      <td>-79.401279</td>
      <td>Mexican Restaurant</td>
      <td>Food</td>
    </tr>
  </tbody>
</table>
</div>



### Check for the correct neighbourhood to each venue and correct if necessary

  The geopandas dataframe has a method to check if a geo-coordinate is with in the boundaries
  of an area, in this case neighbourhood boundaries. The df_toronto_nbh dataframe has a column
  with these boundaries and can be used to verify the venues geo-location.


```python
# loop at all the venues
drop_index_list = []
corrected = 0
for i,ven in df_toronto_ven.iterrows():
    # create a Point based on the venues latitude and longitude coordinates
    pnt = Point(ven['Longitude'],ven['Latitude'])
    # get the venues neighbourhood number
    vcd = ven['CDN_Number']
    # loop at the neighbourhood dataframe
    found = False
    for j, nbh in df_toronto_nbh.iterrows():
        # check if the venues coordinates are within the neighbourhood's boundaries
        isin = pnt.within(nbh['geometry'])
        # the venue is in the current neighbourhood
        if isin:
            found = True
            if vcd != nbh['CDN_Number']:
                # print('Changed')
                corrected = corrected + 1
                df_toronto_ven.at[1,'CDN_Number'] = nbh['CDN_Number']  
                df_toronto_ven.at[1,'Neighbourhood'] = nbh['Neighbourhood']  
            break
    if found == False:
        drop_index_list.append(i)
```

### Report the corrections here and drop any venues that are out of bounds ...


```python
# log the updates and drop rows that are not within any boundaries
print(df_toronto_ven.shape[0], 'venues checked')
# how many venues have had their neighbourhood reassigned
if corrected:
    print(corrected,' venues corrected')
if len(drop_index_list) > 0:
    # drop any rows contained in the drop_index_list => not found
    df_toronto_ven.drop(df_toronto_ven.index[drop_index_list],inplace=True)
    print('Venues removed: ', len(drop_index_list))
    df_toronto_ven.reset_index(drop=True,inplace=True)
# show what is left ...
print(df_toronto_ven.shape[0], 'venues remaining')
```

    3411 venues checked
    1491  venues corrected
    Venues removed:  71
    3340 venues remaining


### Display venues dataframe after assigning the correct neighbourhood
**Note**: As reported almost half of neighbourhood of all venues has been corrected. This is due to the fact
that the Foursquare API endpoint "explore" only accepts a radius from a central point, which can lead to a venue
being outside of the neighbourhood. 76 venues where entirely outside of the neighbourhoods and have been removed


```python
df_toronto_ven.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDN_Number</th>
      <th>Neighbourhood</th>
      <th>Venue</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>SubCategory</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>095</td>
      <td>Annex</td>
      <td>Rose &amp; Sons</td>
      <td>43.675668</td>
      <td>-79.403617</td>
      <td>American Restaurant</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>1</th>
      <td>096</td>
      <td>Casa Loma</td>
      <td>Ezra's Pound</td>
      <td>43.675153</td>
      <td>-79.405858</td>
      <td>Café</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>2</th>
      <td>095</td>
      <td>Annex</td>
      <td>Roti Cuisine of India</td>
      <td>43.674618</td>
      <td>-79.408249</td>
      <td>Indian Restaurant</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>3</th>
      <td>095</td>
      <td>Annex</td>
      <td>Fresh on Bloor</td>
      <td>43.666755</td>
      <td>-79.403491</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>4</th>
      <td>095</td>
      <td>Annex</td>
      <td>Playa Cabana</td>
      <td>43.676112</td>
      <td>-79.401279</td>
      <td>Mexican Restaurant</td>
      <td>Food</td>
    </tr>
  </tbody>
</table>
</div>



### Missing categories

One thing I noticed is that not all the venue subcategories were found according to the Foursquare
categories - subcategories list. Therefore this needs to be corrected as well. Here is where the fun begins
as quite a lot of code is necessary to fix this


```python
# fix the Category column based on certain key words in the subcategory
def fix_category(row):
    #print(pd.isna(row['Category']))
    if pd.isna(row['Category']):
        if 'restaurant' in str(row['SubCategory']).lower():
            return 'Food'
        elif 'food' in str(row['SubCategory']).lower():
            return 'Food'
        elif 'place' in str(row['SubCategory']).lower():
            return 'Food'
        elif 'churrascaria' in str(row['SubCategory']).lower():
            return 'Food'
        elif 'noodle' in str(row['SubCategory']).lower():
            return 'Food'
        elif str(row['SubCategory']) == 'Ice Cream Shop':
            return 'Food'
        elif 'store' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'shop' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'studio' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'gym' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'market' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'butcher' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'boutique' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'grocery' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'dojo' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'chiropractor' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'tech startup' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'coworking space' in str(row['SubCategory']).lower():
            return 'Shop & Service'
        elif 'bar' in str(row['SubCategory']).lower():
            return 'Nightlife Spot'
        elif 'pub' in str(row['SubCategory']).lower():
            return 'Nightlife Spot'
        elif 'club' in str(row['SubCategory']).lower():
            return 'Nightlife Spot'
        elif 'speakeasy' in str(row['SubCategory']).lower():
            return 'Nightlife Spot'
        elif 'theater' in str(row['SubCategory']).lower():
            return 'Arts & Entertainment'
        elif 'museum' in str(row['SubCategory']).lower():
            return 'Arts & Entertainment'
        elif 'bus' in str(row['SubCategory']).lower():
            return 'Travel & Transport'
        elif 'hostel' in str(row['SubCategory']).lower():
            return 'Travel & Transport'
        elif 'platform' in str(row['SubCategory']).lower():
            return 'Travel & Transport'
        elif 'school' in str(row['SubCategory']).lower():
            return 'Professional & Other Places'
        elif 'church' in str(row['SubCategory']).lower():
            return 'Professional & Other Places'
        elif 'field' in str(row['SubCategory']).lower():
            return 'Outdoors & Recreation'
        elif 'court' in str(row['SubCategory']).lower():
            return 'Outdoors & Recreation'
        elif 'track' in str(row['SubCategory']).lower():
            return 'Outdoors & Recreation'
        elif 'rink' in str(row['SubCategory']).lower():
            return 'Outdoors & Recreation'
        elif 'stadium' in str(row['SubCategory']).lower():
            return 'Outdoors & Recreation'
        elif 'monument / landmark' in str(row['SubCategory']).lower():
            return 'Outdoors & Recreation'
        elif 'arena' in str(row['SubCategory']).lower():
            return 'Outdoors & Recreation'
        elif 'curling' in str(row['SubCategory']).lower():
            return 'Outdoors & Recreation'
        elif 'outdoors & recreation' in str(row['SubCategory']).lower():
            return 'Outdoors & Recreation'
        else:
            return row['Category']
    else:
        return row['Category']

```


```python
# fix the venue catagories by first creating a new column and then replacing the old one
df_toronto_ven['New Cat'] = df_toronto_ven.apply(lambda x: fix_category(x),axis=1)
# remove any rows where the subcategory is Neighborhood
df_toronto_ven = df_toronto_ven.query('SubCategory != "Neighborhood"')
# save to csv to check in excel just in case 
df_toronto_ven.to_csv('df_toronto_ven_after.csv')
# do we have any rows left without a category?
df_toronto_ven[df_toronto_ven['New Cat'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDN_Number</th>
      <th>Neighbourhood</th>
      <th>Venue</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>SubCategory</th>
      <th>Category</th>
      <th>New Cat</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### Finally all the venues have been assigned now

One last step to replace the Category column with the "fixed" categories in column "New Cat"


```python
# repair the Category column with the "New Cat" column and then drop "New Cat"
df_toronto_ven['Category'] = df_toronto_ven['New Cat']
df_toronto_ven.drop(columns=['New Cat'],inplace=True)
df_toronto_ven.sort_values(by=['Neighbourhood','Category','SubCategory'],inplace=True)
df_toronto_ven.reset_index(drop=True,inplace=True)
# final look
df_toronto_ven.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDN_Number</th>
      <th>Neighbourhood</th>
      <th>Venue</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>SubCategory</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>095</td>
      <td>Annex</td>
      <td>Koerner Hall</td>
      <td>43.667983</td>
      <td>-79.395962</td>
      <td>Concert Hall</td>
      <td>Arts &amp; Entertainment</td>
    </tr>
    <tr>
      <th>1</th>
      <td>095</td>
      <td>Annex</td>
      <td>Baldwin Steps</td>
      <td>43.677707</td>
      <td>-79.408209</td>
      <td>Historic Site</td>
      <td>Arts &amp; Entertainment</td>
    </tr>
    <tr>
      <th>2</th>
      <td>095</td>
      <td>Annex</td>
      <td>Toronto Archives</td>
      <td>43.676447</td>
      <td>-79.407509</td>
      <td>History Museum</td>
      <td>Arts &amp; Entertainment</td>
    </tr>
    <tr>
      <th>3</th>
      <td>095</td>
      <td>Annex</td>
      <td>The Bloor Hot Docs Cinema</td>
      <td>43.665499</td>
      <td>-79.410313</td>
      <td>Indie Movie Theater</td>
      <td>Arts &amp; Entertainment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>095</td>
      <td>Annex</td>
      <td>Royal Ontario Museum</td>
      <td>43.668367</td>
      <td>-79.394813</td>
      <td>Museum</td>
      <td>Arts &amp; Entertainment</td>
    </tr>
  </tbody>
</table>
</div>



# Data exploration / Methodology <a name="methodology"/>

### Analysis of the data gathered

  To get a general idea, let's see how many venue categories we have found by neighbourhood


```python
nhb_count = len(df_toronto_ven['Neighbourhood'].unique())
sub_count = len(df_toronto_ven['SubCategory'].unique())
cat_count = len(df_toronto_ven['Category'].unique())
ven_count = df_toronto_ven.shape[0]
print('{} top level categories with {} unique venue categories found across {} neighbourhoods\n{} venues in total'.format(
    cat_count,sub_count,nhb_count,ven_count))
```

    8 top level categories with 281 unique venue categories found across 44 neighbourhoods
    3331 venues in total


### Visualize the average household income after tax by neighbourhood

* Build a folium choropleth map of the area to show the average incomes by neighbourhood
* This should give some insight to the analysis of the K-Means clustering further on down


```python
# create map of Toronto Neighbourhoods (FSAs) using retrived latitude and longitude values
map_toronto = folium.Map(location=[43.673963, -79.387207], zoom_start=12);
toronto_geojson = "./data/toronto_neighbourhoods.json"
map_toronto.choropleth(geo_data=toronto_geojson,
    data = df_toronto_nbh,
    popup=df_toronto_nbh['Neighbourhood'],
    columns=['Neighbourhood','AfterTaxHouseholdIncome'],
    key_on='feature.properties.Neighbourhood',
    fill_color='YlOrRd',
    fill_opacity=0.5, 
    line_opacity=0.2,
    legend_name='Average Houseold Income after Tax by Neighbourhood')   
# add markers to map
for lat, lng, cdn_number, neighborhood in zip(df_toronto_nbh['Latitude'], df_toronto_nbh['Longitude'], df_toronto_nbh['CDN_Number'], df_toronto_nbh['Neighbourhood']):
    label = '{} - {}'.format(neighborhood, cdn_number)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=2,
        popup=label,
        color='red',
        fill=True,
        #fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_toronto)  
map_toronto.save('toronto_map_inc.html')
#map_toronto
```

<img src='/images/capstone/toronto_map_inc.jpeg'/>

### Average household income after tax by neighbourhood

The neighbourhoods in the north of Toronto like Lawrence Park South and North are the high income neighbourhoods
Also visible is that neighbourhoods closer to the lakeside have a higher average income as well as those on
the edges of the city. In the central part of Toronto there are neighbourhoods with lesser average income.

For the small businesses within the centeral neighbourhoods of Toronto this doesn't necessarily mean that there
is lesser spending power, as there are potentially more offices in the area. The average neighbourhood income
would not be reflected in the incomes of the people working in these areas.

### Let's have a look at a graph of the population density and after tax income by neighbourhood


```python
sns.set_style('whitegrid')
df_toronto_bar = pd.DataFrame(df_toronto_nbh[['Neighbourhood','PopulationDensity','AfterTaxHouseholdIncome']]).copy()
df_toronto_bar.set_index('Neighbourhood',inplace=True)
fig = df_toronto_bar.plot(kind='bar',figsize=(16,8)).get_figure()
fig.savefig('toronto_inc_bar.png')
plt.show()
```

<img src='/images/capstone/toronto_inc_bar.png'/>


**Note**: Also note that some of the neighbourhoods with a high average income have
a lower population density, which could mean a spacious suburb with large housing plots

### Number of Venues by Neighbourhood
The graph below visualizes the number of venues by neighbourhood. Looking at two of the neighbourhoods
with the highest incomes, Lawrence Park South & North, we notice that the number of venues is relatively
small compared to the others. Forest Hill North & South are similar neighbourhoods.

**Note**: Due to the cap of 100 venues in the Foursquare API endpoint "explore", it is not possible
to retrieve more. 


```python
sns.set_style('whitegrid')
plt.figure(figsize=(12,8))
count_plt = sns.countplot(x="Neighbourhood", data=df_toronto_ven, palette='Greens_d') #,height=12, aspect=0.8)
count_plt.set_ylabel('Number of Venues')
count_plt.set_xticklabels(count_plt.get_xticklabels(), rotation=90)
count_plt.set_title("Number of Venues by Neighbourhood")
plt.show()
count_plt.figure.savefig('toronto_ven_by_nbh.png')
```

<img src='/images/capstone/toronto_ven_by_nbh.png'/>


### Which machine learning algorithm to use?

The goal of for this project was to provide a (future) business owner with business location information
for making a more informed decision. Looking a possible suitable machine learning algorithms, I have chosen to focus on either a recommender system or using K-Means clustering for the solution. 

After long thought on which machine learning algorithm to use, I have decided to use the K-Means
clustering algorithm to provide better insight. Along with the other exploratory data analysis, it should be
possible to categorize the clusters as found by the K-Means clustering algorithm.

### The first step in preparing for the K-Means algorithm:

  * By using the pandas get_dummies method we are creating a dataframe with a column for each category.
  * Then used this dataframe to create a dataframe representing the percentage of venues for a category by neighbourhood


```python
# get the venue category count by neighbourhood to add to the neighbourhoods dataframe
df_toronto_onehot = pd.get_dummies(df_toronto_ven[['Category']], prefix="", prefix_sep="")
# add neighbourhood column back to dataframe
df_toronto_onehot['Neighbourhood'] = df_toronto_ven['Neighbourhood'] 
# add neighbourhood column back to dataframe
df_toronto_onehot['Neighbourhood'] = df_toronto_ven['Neighbourhood'] 
# move neighborhood column to the first column
fixed_columns = [df_toronto_onehot.columns[-1]] + list(df_toronto_onehot.columns[:-1])
df_toronto_onehot = df_toronto_onehot[fixed_columns]
df_toronto_grp = df_toronto_onehot.groupby('Neighbourhood').mean().reset_index()
```

### Add the normalized (between 0 and 1) population denstity and avg. income by neighbourhood ...
  * Add the columns to the df_toronto_grp dataframe after normalizing 


```python
# we need to reset the index before using the dataframe to normalize the attributes
df_toronto_bar.reset_index(inplace=True)
# add the population density and average household income as well and normalize between 0 and 1
# and add to the grouped venues catagory
x = df_toronto_bar[['PopulationDensity','AfterTaxHouseholdIncome']].values #returns a numpy array
# set the range between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
# normalize
x_scaled = min_max_scaler.fit_transform(x) #.reshape(-1,1))
# add the columns to the grouped by category
df_toronto_grp[['PopulationDensity','AfterTaxHouseholdIncome']] = pd.DataFrame(x_scaled,columns=['PopulationDensity','AfterTaxHouseholdIncome'])
df_toronto_grp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Arts &amp; Entertainment</th>
      <th>College &amp; University</th>
      <th>Food</th>
      <th>Nightlife Spot</th>
      <th>Outdoors &amp; Recreation</th>
      <th>Professional &amp; Other Places</th>
      <th>Shop &amp; Service</th>
      <th>Travel &amp; Transport</th>
      <th>PopulationDensity</th>
      <th>AfterTaxHouseholdIncome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Annex</td>
      <td>0.070707</td>
      <td>0.000000</td>
      <td>0.656566</td>
      <td>0.040404</td>
      <td>0.040404</td>
      <td>0.030303</td>
      <td>0.151515</td>
      <td>0.010101</td>
      <td>0.183297</td>
      <td>0.257485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bay Street Corridor</td>
      <td>0.080808</td>
      <td>0.010101</td>
      <td>0.616162</td>
      <td>0.030303</td>
      <td>0.040404</td>
      <td>0.010101</td>
      <td>0.212121</td>
      <td>0.000000</td>
      <td>0.261906</td>
      <td>0.186130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blake-Jones</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.720000</td>
      <td>0.090000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.140000</td>
      <td>0.010000</td>
      <td>0.130220</td>
      <td>0.277270</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cabbagetown-South St.James Town</td>
      <td>0.046512</td>
      <td>0.000000</td>
      <td>0.581395</td>
      <td>0.046512</td>
      <td>0.209302</td>
      <td>0.000000</td>
      <td>0.116279</td>
      <td>0.000000</td>
      <td>0.124467</td>
      <td>0.270428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Casa Loma</td>
      <td>0.068966</td>
      <td>0.000000</td>
      <td>0.609195</td>
      <td>0.011494</td>
      <td>0.068966</td>
      <td>0.011494</td>
      <td>0.195402</td>
      <td>0.034483</td>
      <td>0.065751</td>
      <td>0.468424</td>
    </tr>
  </tbody>
</table>
</div>



### Run the K-Means algorithm several times to determine the optimal number of clusters to use
  * Once run the model returns a value of inertia: model.inertia_.
  * We are looking for a number of clusters where the inertia visibly flattens out


```python
# now get the optimal K
ks = range(2, 14)
inertias = []
df_toronto_grp_clu_tmp = df_toronto_grp.drop('Neighbourhood', 1)

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k,random_state=0)
    
    # Fit model to samples
    model.fit(df_toronto_grp_clu_tmp)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
fig = plt.figure(figsize=(12,8))
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
fig.savefig('kmeans_elbow_diagram.png')
```

<img src='/images/capstone/kmeans_elbow_diagram.png'/>


**Note**: from the elbow graph shown above, the optimal number of clusters is around 7 as the intertia really begins to descrease

### We have determined the optimal number of clusters = 7.

Now run K-Means on the normalized columns of the venues grouped dataframe df_toronto_grp


```python
# set number of clusters as determined in the elbow plot above
kclusters = 7
#df_toronto_grp_clu = df_toronto_grp.drop('Neighbourhood', 1)
# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(df_toronto_grp.drop(columns=['Neighbourhood']))
# check cluster labels generated for each row in the dataframe
kmeans.labels_ 
```




    array([0, 6, 0, 0, 4, 6, 0, 4, 0, 0, 0, 2, 2, 0, 0, 5, 0, 6, 1, 1, 0, 6,
           4, 6, 2, 4, 3, 0, 0, 6, 0, 5, 4, 6, 5, 2, 0, 0, 0, 0, 2, 0, 4, 0],
          dtype=int32)



### Merge the venues grouped by category dataframe along with the neighbourhoods dataframe

  * Add the cluster labels column kmeans.labels_ to the dataframe
  * Merge the venues grouped by category dataframe with the neighbourhoods dataframe
  * Plot the merged dataframe using Choropleth to view the results of the K-Means clustering


```python
df_toronto_grp.insert(loc=0, column='Cluster Labels', value=kmeans.labels_)
# now merge both to one dataframe
df_toronto_mrg = pd.merge(left=df_toronto_nbh,right=df_toronto_grp, on='Neighbourhood')
df_toronto_mrg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDN_Number</th>
      <th>Neighbourhood</th>
      <th>geometry</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>TotalPopulation</th>
      <th>TotalArea</th>
      <th>AfterTaxHouseholdIncome_x</th>
      <th>PopulationDensity_x</th>
      <th>Cluster Labels</th>
      <th>Arts &amp; Entertainment</th>
      <th>College &amp; University</th>
      <th>Food</th>
      <th>Nightlife Spot</th>
      <th>Outdoors &amp; Recreation</th>
      <th>Professional &amp; Other Places</th>
      <th>Shop &amp; Service</th>
      <th>Travel &amp; Transport</th>
      <th>PopulationDensity_y</th>
      <th>AfterTaxHouseholdIncome_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>095</td>
      <td>Annex</td>
      <td>POLYGON ((-79.39414141500001 43.668720261, -79...</td>
      <td>43.671585</td>
      <td>-79.404000</td>
      <td>30526</td>
      <td>2.8</td>
      <td>49912</td>
      <td>10902.0</td>
      <td>0</td>
      <td>0.070707</td>
      <td>0.000000</td>
      <td>0.656566</td>
      <td>0.040404</td>
      <td>0.040404</td>
      <td>0.030303</td>
      <td>0.151515</td>
      <td>0.010101</td>
      <td>0.183297</td>
      <td>0.257485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>076</td>
      <td>Bay Street Corridor</td>
      <td>POLYGON ((-79.38751633 43.650672917, -79.38662...</td>
      <td>43.657512</td>
      <td>-79.385722</td>
      <td>25797</td>
      <td>1.8</td>
      <td>44614</td>
      <td>14332.0</td>
      <td>6</td>
      <td>0.080808</td>
      <td>0.010101</td>
      <td>0.616162</td>
      <td>0.030303</td>
      <td>0.040404</td>
      <td>0.010101</td>
      <td>0.212121</td>
      <td>0.000000</td>
      <td>0.261906</td>
      <td>0.186130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>069</td>
      <td>Blake-Jones</td>
      <td>POLYGON ((-79.34082169200001 43.669213123, -79...</td>
      <td>43.676173</td>
      <td>-79.337394</td>
      <td>7727</td>
      <td>0.9</td>
      <td>51381</td>
      <td>8586.0</td>
      <td>0</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.720000</td>
      <td>0.090000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.140000</td>
      <td>0.010000</td>
      <td>0.130220</td>
      <td>0.277270</td>
    </tr>
    <tr>
      <th>3</th>
      <td>071</td>
      <td>Cabbagetown-South St.James Town</td>
      <td>POLYGON ((-79.376716938 43.662418858, -79.3772...</td>
      <td>43.667648</td>
      <td>-79.366107</td>
      <td>11669</td>
      <td>1.4</td>
      <td>50873</td>
      <td>8335.0</td>
      <td>0</td>
      <td>0.046512</td>
      <td>0.000000</td>
      <td>0.581395</td>
      <td>0.046512</td>
      <td>0.209302</td>
      <td>0.000000</td>
      <td>0.116279</td>
      <td>0.000000</td>
      <td>0.124467</td>
      <td>0.270428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>096</td>
      <td>Casa Loma</td>
      <td>POLYGON ((-79.414693177 43.673910413, -79.4148...</td>
      <td>43.681852</td>
      <td>-79.408007</td>
      <td>10968</td>
      <td>1.9</td>
      <td>65574</td>
      <td>5773.0</td>
      <td>4</td>
      <td>0.068966</td>
      <td>0.000000</td>
      <td>0.609195</td>
      <td>0.011494</td>
      <td>0.068966</td>
      <td>0.011494</td>
      <td>0.195402</td>
      <td>0.034483</td>
      <td>0.065751</td>
      <td>0.468424</td>
    </tr>
  </tbody>
</table>
</div>



## Analysis <a name="analysis"/>

### Visualize the K-Means clustering by neighbourhood on a map

  * Map the outlines of the neighbourhood boundaries
  * Plot the assigned cluster of each neighbourhood using a different color for each cluster
  * With this plot is should be easier to discover the patterns in the clustering assignment


```python
import matplotlib.cm as cm
import matplotlib.colors as colors
# create map
map_toronto_clu = folium.Map(location=[43.673963, -79.387207], zoom_start=12)
# draw boundaries
map_toronto_clu.choropleth(geo_data=toronto_geojson,
      fill_opacity=0.1,
      line_opacity=0.5,
      legend_name='K-Means Clusters by Neighbourhood')   
# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]
rainbow = ['#9400D3','#4B0082','#0000FF','#00FF00','#FFFF00','#FF7F00','#FF0000']

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(df_toronto_mrg['Latitude'], df_toronto_mrg['Longitude'], df_toronto_mrg['Neighbourhood'], df_toronto_mrg['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    toolt = str(poi) + ' Cluster ' + str(cluster)
    folium.CircleMarker(
        [lat, lon],
        radius=6,
        popup=label,
        tooltip=toolt,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.3).add_to(map_toronto_clu) 
map_toronto_clu.save('toronto_map_clu.html')
#map_toronto_clu
```

<img src="/images/capstone/toronto_map_clu2.jpeg"/>

**Legend**: (including initial analysis on the clustering results after looking at the map)
  1. Red = Cluster 0<br>- Most prominent, grouped around the downtown area
  2. Light Purple = Cluster 1<br>- Most northern neighbourhoods with a low population density and high avg. income
  3. Dark Purple = Cluster 2<br>- Appear to be located on outer neighbourhoods of the research area or close to a recreational area, needs further investigation
  4. Blue = Cluster 3<br>- Only one neighbourhood represented, needs some further investigation
  5. Green = Cluster 4<br>- Also appear to be located on outer neighbourhoods of the research area
  6. Yellow = Cluster 5<br>- Seems to be neighbourhoods with a larger park or recreational facilities
  7. Orange = Cluster 6<br>- Mainly grouped in the downtown area

### Analyze the clusters by looking at most prominent venue categories by neighbourhood
  * Create a cross table dataframe with number of venues by category by neighbourhood
  * We can use this to create a stacked bar plot showing the proportion of venues by category for each neighbourhood
  * This plot should also be helpful in detecting patterns behind the clustering assignment


```python
# get the number of venues by category by neighbourhood
df_toronto_grp_cnt = df_toronto_ven.groupby(by=['Neighbourhood','Category']).size().unstack(fill_value=0)
df_toronto_grp_cnt.insert(loc=0, column='Cluster Labels', value=kmeans.labels_)
df_toronto_grp_cnt.sort_values(by=['Cluster Labels','Neighbourhood'],inplace=True)
df_toronto_grp_cnt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>Cluster Labels</th>
      <th>Arts &amp; Entertainment</th>
      <th>College &amp; University</th>
      <th>Food</th>
      <th>Nightlife Spot</th>
      <th>Outdoors &amp; Recreation</th>
      <th>Professional &amp; Other Places</th>
      <th>Shop &amp; Service</th>
      <th>Travel &amp; Transport</th>
    </tr>
    <tr>
      <th>Neighbourhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Annex</th>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>65</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Blake-Jones</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>72</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Cabbagetown-South St.James Town</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>25</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Corso Italia-Davenport</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Dovercourt-Wallace Emerson-Junction</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>54</td>
      <td>18</td>
      <td>4</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Plot the number of venues by category by neighbourhood


```python
# plot the number of venues by category in a stacked bar plot by neighbourhood
wid_nbh = 0.5
cum_val = 0
# we want a list of column names (categories) sorted by the most venues in a category in descending order
col_nams = list(df_toronto_grp_cnt.sum().sort_values(ascending=False).index)
col_nams.remove('Cluster Labels')
fig = plt.figure(figsize=(14,12))
for col in col_nams:
    plt.bar(list(df_toronto_grp_cnt.index), df_toronto_grp_cnt[col], bottom=cum_val, label=col)
    cum_val = cum_val+df_toronto_grp_cnt[col]
_ = plt.xticks(rotation=90)
_ = plt.yticks(np.arange(0, 120, 10))
_ = plt.legend(fontsize=10)
plt.show()
fig.savefig('toronto_venues_by_nbh.png')
```

<img src='/images/capstone/toronto_venues_by_nbh.png'/>


### Several observations here:
  **Neighbourhoods by cluster label number**:
  0. **From Annex to Wychwood**<br>
  Cluster 2 neighbourhoods are in the majority. They have a relatively large number of food related venues<br>
  And shops, services and nightlife venues are also prominent. The locations are located to the west/north-west<br>
  of the downtown area, as well as to the east
  1. **Lawrence Park North and South**<br>
  Both neighbourhoods have a high average income and low population density which would lead to<br>
  conclude that these are mainly residential areas
  2. **Forest Hill North to Woodbine Corridor**<br>
  There appears to be a relatively large number of shops and services in these neighbourhoods<br>
  And also relatively less nightlife spots (bars, clubs etc.) compared to the cluster 0 (red) neighbourhoods.<br>
  The outdoors and recreational venues are also prominent. 
  3. **North St.James Town**<br>
  This neighbourhood has been separately clustered due to the fact that there it has the highest population<br>
  density of all the researched negighbourhoods. It has a relatively high number of shops and service venues.
  4. **From Casa Loma to Yonge-Eglinton**
  The cluster 4 neighbourhoods are located in the north and eastern part of the research area with one exception<br>
  located in the far west. Mainly on the outskirts. There are a relatively large number of shops and services<br>
  as well as outdoor and recreational venues. Nightlife venues are also prominent.
  5. **High Park-Swansea to South Riverdale**<br>
  These neighbourhoods are located in recreational areas like parks or close to the waterfront and have a<br>
  high percentage of outdoor and recreational venues as well as travel and transportation venues<br>
  (hotels, transportation hubs: bus, metro or train stations)
  6. **From Bay Street Corridor to South Parkdale**<br>
  These are neighbourhoods in the downtown area of Toronto with a high number of venues in the food category<br>
  like restaurants. Shops and services are also prominent as well as venues in the nightlife spot category
  
  
  * In almost all neighbourhoods venues within the food category (restaurants, coffee shops etc.)<br>
  are the most prominent
  * Shops & Services are the second most prominent all-round (stores, shops, fitness studios etc.)
  * Nightlife Spots are the third most prominent all-round (bars, speakeasy's, clubs etc)
  

### Create a dataframe to visualize the venue categories by venue count by neighbourhood


```python
# create a sorted list for each neighbourhood with the category with highest number of venues first
# loop at the neighbourhoods 
nbh_dict = dict()
for nbh,row in df_toronto_grp_cnt.iterrows():
    ven_arr = []
    for ven_cat in df_toronto_grp_cnt.drop(columns='Cluster Labels').columns:
        ven_cnt = row[ven_cat]
        ven_arr.append([ven_cat,ven_cnt])
    # sort the array by the number of venues by category
    ven_arr = sorted(ven_arr, key=lambda x: x[1], reverse=True)
    # flatten the array to one dimension
    flat_arr = [val for sublist in ven_arr for val in sublist]
    # create a dictionary entry for the current neighbourhood
    nbh_dict[row.name] = flat_arr
# switch the column names and index around
df_toronto_grp_cnt_srt = pd.DataFrame(nbh_dict).transpose()
# now we need to fix the column headers to readable text
indic = ['st', 'nd', 'rd']
# create columns according to number of venues by category
columns = []
cnt = 0
for i in np.arange(len(df_toronto_grp_cnt_srt.columns)):
    if i % 2 == 0:
        cnt = cnt + 1
        try:
            columns.append('{}{} Category'.format(cnt, indic[cnt-1]))
        except:
            columns.append('{}th Category'.format(cnt))
    else:
        try:
            columns.append('{}{} # Venues'.format(cnt, indic[cnt-1]))
        except:
            columns.append('{}th # Venues'.format(cnt))
df_toronto_grp_cnt_srt.columns = columns
# we need to move the index to a column 
df_toronto_grp_cnt_srt.reset_index(inplace=True)
df_toronto_grp_cnt_srt.rename(columns={'index':'Neighbourhood'},inplace=True)
df_toronto_grp_cnt_srt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>1st Category</th>
      <th>1st # Venues</th>
      <th>2nd Category</th>
      <th>2nd # Venues</th>
      <th>3rd Category</th>
      <th>3rd # Venues</th>
      <th>4th Category</th>
      <th>4th # Venues</th>
      <th>5th Category</th>
      <th>5th # Venues</th>
      <th>6th Category</th>
      <th>6th # Venues</th>
      <th>7th Category</th>
      <th>7th # Venues</th>
      <th>8th Category</th>
      <th>8th # Venues</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Annex</td>
      <td>Food</td>
      <td>65</td>
      <td>Shop &amp; Service</td>
      <td>15</td>
      <td>Arts &amp; Entertainment</td>
      <td>7</td>
      <td>Nightlife Spot</td>
      <td>4</td>
      <td>Outdoors &amp; Recreation</td>
      <td>4</td>
      <td>Professional &amp; Other Places</td>
      <td>3</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blake-Jones</td>
      <td>Food</td>
      <td>72</td>
      <td>Shop &amp; Service</td>
      <td>14</td>
      <td>Nightlife Spot</td>
      <td>9</td>
      <td>Arts &amp; Entertainment</td>
      <td>2</td>
      <td>Outdoors &amp; Recreation</td>
      <td>2</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cabbagetown-South St.James Town</td>
      <td>Food</td>
      <td>25</td>
      <td>Outdoors &amp; Recreation</td>
      <td>9</td>
      <td>Shop &amp; Service</td>
      <td>5</td>
      <td>Arts &amp; Entertainment</td>
      <td>2</td>
      <td>Nightlife Spot</td>
      <td>2</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Corso Italia-Davenport</td>
      <td>Food</td>
      <td>30</td>
      <td>Shop &amp; Service</td>
      <td>6</td>
      <td>Outdoors &amp; Recreation</td>
      <td>3</td>
      <td>Travel &amp; Transport</td>
      <td>3</td>
      <td>Nightlife Spot</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dovercourt-Wallace Emerson-Junction</td>
      <td>Food</td>
      <td>54</td>
      <td>Shop &amp; Service</td>
      <td>20</td>
      <td>Nightlife Spot</td>
      <td>18</td>
      <td>Outdoors &amp; Recreation</td>
      <td>4</td>
      <td>Arts &amp; Entertainment</td>
      <td>3</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### We will use this dataframe to merge with the clusters by neighbourhoods dataframe to have one dataframe for analysis


```python
# add the cluster number, population density and average income so we can do some analysis on the clusters
df_toronto_analize = pd.merge(left=df_toronto_mrg[['Neighbourhood','Cluster Labels','AfterTaxHouseholdIncome_x','PopulationDensity_x']],
                              right=df_toronto_grp_cnt_srt,
                              on='Neighbourhood')
df_toronto_analize.rename(columns={'Cluster Labels':'Cluster','AfterTaxHouseholdIncome_x':'AvgIncome','PopulationDensity_x':'PopDensity'},inplace=True)
# display neighbourhoods by cluster
df_toronto_analize.sort_values(by=['Cluster','Neighbourhood']).to_csv('df_toronto_analize.csv')
df_toronto_analize.sort_values(by=['Cluster','Neighbourhood'],inplace=True)
df_toronto_analize
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Cluster</th>
      <th>AvgIncome</th>
      <th>PopDensity</th>
      <th>1st Category</th>
      <th>1st # Venues</th>
      <th>2nd Category</th>
      <th>2nd # Venues</th>
      <th>3rd Category</th>
      <th>3rd # Venues</th>
      <th>4th Category</th>
      <th>4th # Venues</th>
      <th>5th Category</th>
      <th>5th # Venues</th>
      <th>6th Category</th>
      <th>6th # Venues</th>
      <th>7th Category</th>
      <th>7th # Venues</th>
      <th>8th Category</th>
      <th>8th # Venues</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Annex</td>
      <td>0</td>
      <td>49912</td>
      <td>10902.0</td>
      <td>Food</td>
      <td>65</td>
      <td>Shop &amp; Service</td>
      <td>15</td>
      <td>Arts &amp; Entertainment</td>
      <td>7</td>
      <td>Nightlife Spot</td>
      <td>4</td>
      <td>Outdoors &amp; Recreation</td>
      <td>4</td>
      <td>Professional &amp; Other Places</td>
      <td>3</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blake-Jones</td>
      <td>0</td>
      <td>51381</td>
      <td>8586.0</td>
      <td>Food</td>
      <td>72</td>
      <td>Shop &amp; Service</td>
      <td>14</td>
      <td>Nightlife Spot</td>
      <td>9</td>
      <td>Arts &amp; Entertainment</td>
      <td>2</td>
      <td>Outdoors &amp; Recreation</td>
      <td>2</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cabbagetown-South St.James Town</td>
      <td>0</td>
      <td>50873</td>
      <td>8335.0</td>
      <td>Food</td>
      <td>25</td>
      <td>Outdoors &amp; Recreation</td>
      <td>9</td>
      <td>Shop &amp; Service</td>
      <td>5</td>
      <td>Arts &amp; Entertainment</td>
      <td>2</td>
      <td>Nightlife Spot</td>
      <td>2</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Corso Italia-Davenport</td>
      <td>0</td>
      <td>56345</td>
      <td>7438.0</td>
      <td>Food</td>
      <td>30</td>
      <td>Shop &amp; Service</td>
      <td>6</td>
      <td>Outdoors &amp; Recreation</td>
      <td>3</td>
      <td>Travel &amp; Transport</td>
      <td>3</td>
      <td>Nightlife Spot</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dovercourt-Wallace Emerson-Junction</td>
      <td>0</td>
      <td>50741</td>
      <td>9899.0</td>
      <td>Food</td>
      <td>54</td>
      <td>Shop &amp; Service</td>
      <td>20</td>
      <td>Nightlife Spot</td>
      <td>18</td>
      <td>Outdoors &amp; Recreation</td>
      <td>4</td>
      <td>Arts &amp; Entertainment</td>
      <td>3</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dufferin Grove</td>
      <td>0</td>
      <td>44145</td>
      <td>8418.0</td>
      <td>Food</td>
      <td>37</td>
      <td>Nightlife Spot</td>
      <td>11</td>
      <td>Shop &amp; Service</td>
      <td>5</td>
      <td>Outdoors &amp; Recreation</td>
      <td>4</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>East End-Danforth</td>
      <td>0</td>
      <td>56179</td>
      <td>8223.0</td>
      <td>Food</td>
      <td>33</td>
      <td>Outdoors &amp; Recreation</td>
      <td>5</td>
      <td>Shop &amp; Service</td>
      <td>5</td>
      <td>Travel &amp; Transport</td>
      <td>5</td>
      <td>Nightlife Spot</td>
      <td>3</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Greenwood-Coxwell</td>
      <td>0</td>
      <td>52770</td>
      <td>8481.0</td>
      <td>Food</td>
      <td>36</td>
      <td>Shop &amp; Service</td>
      <td>9</td>
      <td>Outdoors &amp; Recreation</td>
      <td>6</td>
      <td>Nightlife Spot</td>
      <td>3</td>
      <td>Arts &amp; Entertainment</td>
      <td>2</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>High Park North</td>
      <td>0</td>
      <td>52827</td>
      <td>11664.0</td>
      <td>Food</td>
      <td>51</td>
      <td>Shop &amp; Service</td>
      <td>22</td>
      <td>Nightlife Spot</td>
      <td>11</td>
      <td>Outdoors &amp; Recreation</td>
      <td>5</td>
      <td>Arts &amp; Entertainment</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Junction Area</td>
      <td>0</td>
      <td>53804</td>
      <td>5525.0</td>
      <td>Food</td>
      <td>56</td>
      <td>Shop &amp; Service</td>
      <td>30</td>
      <td>Nightlife Spot</td>
      <td>8</td>
      <td>Outdoors &amp; Recreation</td>
      <td>3</td>
      <td>Travel &amp; Transport</td>
      <td>2</td>
      <td>Arts &amp; Entertainment</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Little Portugal</td>
      <td>0</td>
      <td>52519</td>
      <td>12966.0</td>
      <td>Food</td>
      <td>63</td>
      <td>Nightlife Spot</td>
      <td>20</td>
      <td>Shop &amp; Service</td>
      <td>10</td>
      <td>Arts &amp; Entertainment</td>
      <td>3</td>
      <td>Outdoors &amp; Recreation</td>
      <td>2</td>
      <td>Travel &amp; Transport</td>
      <td>2</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Palmerston-Little Italy</td>
      <td>0</td>
      <td>52309</td>
      <td>9876.0</td>
      <td>Food</td>
      <td>63</td>
      <td>Nightlife Spot</td>
      <td>18</td>
      <td>Shop &amp; Service</td>
      <td>14</td>
      <td>Arts &amp; Entertainment</td>
      <td>4</td>
      <td>Outdoors &amp; Recreation</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Playter Estates-Danforth</td>
      <td>0</td>
      <td>55536</td>
      <td>8671.0</td>
      <td>Food</td>
      <td>54</td>
      <td>Shop &amp; Service</td>
      <td>20</td>
      <td>Nightlife Spot</td>
      <td>7</td>
      <td>Outdoors &amp; Recreation</td>
      <td>4</td>
      <td>Arts &amp; Entertainment</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Roncesvalles</td>
      <td>0</td>
      <td>46883</td>
      <td>9983.0</td>
      <td>Food</td>
      <td>71</td>
      <td>Shop &amp; Service</td>
      <td>17</td>
      <td>Nightlife Spot</td>
      <td>8</td>
      <td>Arts &amp; Entertainment</td>
      <td>2</td>
      <td>Outdoors &amp; Recreation</td>
      <td>2</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Trinity-Bellwoods</td>
      <td>0</td>
      <td>51502</td>
      <td>9739.0</td>
      <td>Food</td>
      <td>57</td>
      <td>Nightlife Spot</td>
      <td>21</td>
      <td>Shop &amp; Service</td>
      <td>15</td>
      <td>Arts &amp; Entertainment</td>
      <td>5</td>
      <td>Outdoors &amp; Recreation</td>
      <td>2</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>University</td>
      <td>0</td>
      <td>45538</td>
      <td>5434.0</td>
      <td>Food</td>
      <td>61</td>
      <td>Shop &amp; Service</td>
      <td>18</td>
      <td>Nightlife Spot</td>
      <td>10</td>
      <td>Arts &amp; Entertainment</td>
      <td>6</td>
      <td>Outdoors &amp; Recreation</td>
      <td>2</td>
      <td>College &amp; University</td>
      <td>1</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Waterfront Communities-The Island</td>
      <td>0</td>
      <td>57670</td>
      <td>8673.0</td>
      <td>Food</td>
      <td>44</td>
      <td>Travel &amp; Transport</td>
      <td>11</td>
      <td>Outdoors &amp; Recreation</td>
      <td>9</td>
      <td>Nightlife Spot</td>
      <td>8</td>
      <td>Shop &amp; Service</td>
      <td>7</td>
      <td>Arts &amp; Entertainment</td>
      <td>5</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Weston-Pellam Park</td>
      <td>0</td>
      <td>48206</td>
      <td>7399.0</td>
      <td>Food</td>
      <td>27</td>
      <td>Shop &amp; Service</td>
      <td>19</td>
      <td>Outdoors &amp; Recreation</td>
      <td>3</td>
      <td>Nightlife Spot</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Wychwood</td>
      <td>0</td>
      <td>50261</td>
      <td>8441.0</td>
      <td>Food</td>
      <td>55</td>
      <td>Shop &amp; Service</td>
      <td>22</td>
      <td>Nightlife Spot</td>
      <td>3</td>
      <td>Outdoors &amp; Recreation</td>
      <td>3</td>
      <td>Arts &amp; Entertainment</td>
      <td>2</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Yonge-St.Clair</td>
      <td>0</td>
      <td>58838</td>
      <td>10440.0</td>
      <td>Food</td>
      <td>45</td>
      <td>Shop &amp; Service</td>
      <td>19</td>
      <td>Nightlife Spot</td>
      <td>5</td>
      <td>Outdoors &amp; Recreation</td>
      <td>5</td>
      <td>Travel &amp; Transport</td>
      <td>2</td>
      <td>Arts &amp; Entertainment</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Lawrence Park North</td>
      <td>1</td>
      <td>103660</td>
      <td>6351.0</td>
      <td>Food</td>
      <td>37</td>
      <td>Shop &amp; Service</td>
      <td>12</td>
      <td>Nightlife Spot</td>
      <td>2</td>
      <td>Travel &amp; Transport</td>
      <td>2</td>
      <td>Outdoors &amp; Recreation</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lawrence Park South</td>
      <td>1</td>
      <td>105043</td>
      <td>4743.0</td>
      <td>Food</td>
      <td>16</td>
      <td>Shop &amp; Service</td>
      <td>11</td>
      <td>Outdoors &amp; Recreation</td>
      <td>4</td>
      <td>Nightlife Spot</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Forest Hill North</td>
      <td>2</td>
      <td>53978</td>
      <td>8004.0</td>
      <td>Food</td>
      <td>11</td>
      <td>Outdoors &amp; Recreation</td>
      <td>6</td>
      <td>Shop &amp; Service</td>
      <td>2</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Nightlife Spot</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Forest Hill South</td>
      <td>2</td>
      <td>67446</td>
      <td>4293.0</td>
      <td>Food</td>
      <td>17</td>
      <td>Shop &amp; Service</td>
      <td>9</td>
      <td>Outdoors &amp; Recreation</td>
      <td>6</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Nightlife Spot</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Niagara</td>
      <td>2</td>
      <td>59929</td>
      <td>10058.0</td>
      <td>Food</td>
      <td>46</td>
      <td>Shop &amp; Service</td>
      <td>27</td>
      <td>Outdoors &amp; Recreation</td>
      <td>13</td>
      <td>Arts &amp; Entertainment</td>
      <td>9</td>
      <td>Nightlife Spot</td>
      <td>3</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>The Beaches</td>
      <td>2</td>
      <td>70957</td>
      <td>5991.0</td>
      <td>Food</td>
      <td>41</td>
      <td>Shop &amp; Service</td>
      <td>19</td>
      <td>Outdoors &amp; Recreation</td>
      <td>10</td>
      <td>Nightlife Spot</td>
      <td>8</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Woodbine Corridor</td>
      <td>2</td>
      <td>63343</td>
      <td>7838.0</td>
      <td>Food</td>
      <td>38</td>
      <td>Shop &amp; Service</td>
      <td>11</td>
      <td>Outdoors &amp; Recreation</td>
      <td>8</td>
      <td>Nightlife Spot</td>
      <td>4</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>North St.James Town</td>
      <td>3</td>
      <td>31304</td>
      <td>46538.0</td>
      <td>Food</td>
      <td>56</td>
      <td>Shop &amp; Service</td>
      <td>26</td>
      <td>Nightlife Spot</td>
      <td>9</td>
      <td>Arts &amp; Entertainment</td>
      <td>4</td>
      <td>Outdoors &amp; Recreation</td>
      <td>4</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Casa Loma</td>
      <td>4</td>
      <td>65574</td>
      <td>5773.0</td>
      <td>Food</td>
      <td>53</td>
      <td>Shop &amp; Service</td>
      <td>17</td>
      <td>Arts &amp; Entertainment</td>
      <td>6</td>
      <td>Outdoors &amp; Recreation</td>
      <td>6</td>
      <td>Travel &amp; Transport</td>
      <td>3</td>
      <td>Nightlife Spot</td>
      <td>1</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Danforth</td>
      <td>4</td>
      <td>62482</td>
      <td>8787.0</td>
      <td>Food</td>
      <td>40</td>
      <td>Shop &amp; Service</td>
      <td>9</td>
      <td>Nightlife Spot</td>
      <td>6</td>
      <td>Outdoors &amp; Recreation</td>
      <td>5</td>
      <td>Arts &amp; Entertainment</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Mount Pleasant East</td>
      <td>4</td>
      <td>71154</td>
      <td>5411.0</td>
      <td>Food</td>
      <td>66</td>
      <td>Shop &amp; Service</td>
      <td>15</td>
      <td>Nightlife Spot</td>
      <td>3</td>
      <td>Outdoors &amp; Recreation</td>
      <td>3</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>North Riverdale</td>
      <td>4</td>
      <td>68164</td>
      <td>6620.0</td>
      <td>Food</td>
      <td>70</td>
      <td>Shop &amp; Service</td>
      <td>15</td>
      <td>Nightlife Spot</td>
      <td>7</td>
      <td>Outdoors &amp; Recreation</td>
      <td>6</td>
      <td>Arts &amp; Entertainment</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Runnymede-Bloor West Village</td>
      <td>4</td>
      <td>74729</td>
      <td>6294.0</td>
      <td>Food</td>
      <td>14</td>
      <td>Shop &amp; Service</td>
      <td>4</td>
      <td>Nightlife Spot</td>
      <td>2</td>
      <td>Outdoors &amp; Recreation</td>
      <td>1</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Yonge-Eglinton</td>
      <td>4</td>
      <td>63267</td>
      <td>7386.0</td>
      <td>Food</td>
      <td>74</td>
      <td>Shop &amp; Service</td>
      <td>15</td>
      <td>Outdoors &amp; Recreation</td>
      <td>6</td>
      <td>Nightlife Spot</td>
      <td>3</td>
      <td>Arts &amp; Entertainment</td>
      <td>2</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>High Park-Swansea</td>
      <td>5</td>
      <td>62128</td>
      <td>4514.0</td>
      <td>Food</td>
      <td>17</td>
      <td>Outdoors &amp; Recreation</td>
      <td>16</td>
      <td>Shop &amp; Service</td>
      <td>13</td>
      <td>Travel &amp; Transport</td>
      <td>5</td>
      <td>Arts &amp; Entertainment</td>
      <td>2</td>
      <td>Nightlife Spot</td>
      <td>2</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Rosedale-Moore Park</td>
      <td>5</td>
      <td>72915</td>
      <td>4548.0</td>
      <td>Food</td>
      <td>15</td>
      <td>Shop &amp; Service</td>
      <td>15</td>
      <td>Outdoors &amp; Recreation</td>
      <td>10</td>
      <td>Nightlife Spot</td>
      <td>2</td>
      <td>Arts &amp; Entertainment</td>
      <td>0</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>South Riverdale</td>
      <td>5</td>
      <td>56192</td>
      <td>2904.0</td>
      <td>Outdoors &amp; Recreation</td>
      <td>5</td>
      <td>Shop &amp; Service</td>
      <td>3</td>
      <td>Food</td>
      <td>2</td>
      <td>Arts &amp; Entertainment</td>
      <td>1</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Nightlife Spot</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bay Street Corridor</td>
      <td>6</td>
      <td>44614</td>
      <td>14332.0</td>
      <td>Food</td>
      <td>61</td>
      <td>Shop &amp; Service</td>
      <td>21</td>
      <td>Arts &amp; Entertainment</td>
      <td>8</td>
      <td>Outdoors &amp; Recreation</td>
      <td>4</td>
      <td>Nightlife Spot</td>
      <td>3</td>
      <td>College &amp; University</td>
      <td>1</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Church-Yonge Corridor</td>
      <td>6</td>
      <td>41813</td>
      <td>22386.0</td>
      <td>Food</td>
      <td>58</td>
      <td>Shop &amp; Service</td>
      <td>22</td>
      <td>Nightlife Spot</td>
      <td>9</td>
      <td>Arts &amp; Entertainment</td>
      <td>6</td>
      <td>Outdoors &amp; Recreation</td>
      <td>3</td>
      <td>College &amp; University</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kensington-Chinatown</td>
      <td>6</td>
      <td>37571</td>
      <td>11963.0</td>
      <td>Food</td>
      <td>60</td>
      <td>Shop &amp; Service</td>
      <td>23</td>
      <td>Nightlife Spot</td>
      <td>10</td>
      <td>Arts &amp; Entertainment</td>
      <td>4</td>
      <td>College &amp; University</td>
      <td>1</td>
      <td>Outdoors &amp; Recreation</td>
      <td>1</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Moss Park</td>
      <td>6</td>
      <td>37295</td>
      <td>14647.0</td>
      <td>Food</td>
      <td>63</td>
      <td>Shop &amp; Service</td>
      <td>13</td>
      <td>Arts &amp; Entertainment</td>
      <td>8</td>
      <td>Nightlife Spot</td>
      <td>6</td>
      <td>Outdoors &amp; Recreation</td>
      <td>6</td>
      <td>Professional &amp; Other Places</td>
      <td>3</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Mount Pleasant West</td>
      <td>6</td>
      <td>48066</td>
      <td>22814.0</td>
      <td>Food</td>
      <td>49</td>
      <td>Shop &amp; Service</td>
      <td>10</td>
      <td>Outdoors &amp; Recreation</td>
      <td>7</td>
      <td>Nightlife Spot</td>
      <td>3</td>
      <td>Arts &amp; Entertainment</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Professional &amp; Other Places</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Regent Park</td>
      <td>6</td>
      <td>30794</td>
      <td>18005.0</td>
      <td>Food</td>
      <td>62</td>
      <td>Shop &amp; Service</td>
      <td>16</td>
      <td>Nightlife Spot</td>
      <td>8</td>
      <td>Outdoors &amp; Recreation</td>
      <td>7</td>
      <td>Arts &amp; Entertainment</td>
      <td>3</td>
      <td>Professional &amp; Other Places</td>
      <td>2</td>
      <td>Travel &amp; Transport</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>South Parkdale</td>
      <td>6</td>
      <td>32539</td>
      <td>9500.0</td>
      <td>Food</td>
      <td>50</td>
      <td>Shop &amp; Service</td>
      <td>22</td>
      <td>Nightlife Spot</td>
      <td>9</td>
      <td>Outdoors &amp; Recreation</td>
      <td>6</td>
      <td>Arts &amp; Entertainment</td>
      <td>1</td>
      <td>Professional &amp; Other Places</td>
      <td>1</td>
      <td>College &amp; University</td>
      <td>0</td>
      <td>Travel &amp; Transport</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Look a the mean and medians of average income and population density


```python
# look a the mean and medians of average income and population density
df_toronto_analize[['AvgIncome','PopDensity']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AvgIncome</th>
      <th>PopDensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>44.000000</td>
      <td>44.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>55981.727273</td>
      <td>9972.568182</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15130.504763</td>
      <td>7034.026401</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30794.000000</td>
      <td>2904.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>48171.000000</td>
      <td>6336.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>53315.500000</td>
      <td>8461.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>62678.250000</td>
      <td>10153.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>105043.000000</td>
      <td>46538.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Analysis of clusters and neighbourhoods dataframe
- **Cluster 0** neighbourhoods have an average household income around the median of 53,315 Canadian dollars.
- **Cluster 1** neighbourhoods are those with the highest average household income and a low population density.<br> This would indicate a rich residential area 
- **Cluster 2** neighbourhoods have a higher than average household income above the overal median of 53,315 Canadian dollars.<br> The population density is lower than the overall median populatin density.
- **Cluster 3** neighbourhood North St.James Town has a very high population density and much lower than median household income.
- **Cluster 4** neighbourhoods have a relatively high average income and lower than median population density<br> which leads me to believe that these are mainly residential areas
- **Cluster 5** neighbourhoods High Park-Swansea, Rosedale-Moore Park and South Riverdale have a higher than median average household income<br> and a much lower than median population density. This is due to the fact that these neighbourhood all contain larger parks or beach areas within their boundaries.
- **Cluster 6** neighbourhoods have a relatively low average household income compared to the overall median average income of 53,351 Canadian dollars. The population density is high. This is most likely due to the fact that these neighbourhoods are located in the downtown area where the real estate prices are high leading to more concentration by square kilometer. 

