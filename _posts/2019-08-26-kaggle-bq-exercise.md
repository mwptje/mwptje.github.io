---
title: "Stocking Rental Bikes"
date: 2019-08-26
tags: [data science, bigquery, python, Kaggle]
header:
  image: "/images/kagggle-bq-exercise_files/baden_baden_panorama_2.jpg"
excerpt: "Data Science, Google Cloud, BigQuery, Python, Kaggle"
mathjax: "true"
---

<img src="https://images.unsplash.com/photo-1559835557-7766a856e6ee?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2100&q=80" width="800" height="600" />

**Photo by Kelly Sikkema on Unsplash**

This exercise is based on Rachel Tatman's tutorial on GCP's BigQuery machine learning option [tutorial](https://www.kaggle.com/rtatman/bigquery-machine-learning-tutorial). 
 
The exercise that goes along with it can be found [here](https://www.kaggle.com/rtatman/bigquery-machine-learning-exercise).

The problem to solve is the following:<br>
"**You stock bikes for a bike rental company in Austin, ensuring stations have enough bikes for all their riders. You want to build a model to predict how many riders will start from each station during each hour, capturing patterns in seasonality, time of day, day of the week, etc.**"

To solve this a kaggle kernel was created using Googles BigQuery in order to explore the data provided and to create a model using linear regression to predict bicycle usage by bike shareing station.

To get started, create a project in GCP and connect to it by running the code cell below. Make sure you have connected the kernel to your GCP account in Settings by enabling BigQuery.

```python
# Set your own project id here
# as a string, the name of the BigQuery project you have created beforehand
PROJECT_ID = 'kaggle-bq-mwptje-exercise'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('model_dataset', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID
```

### Load Google Cloud Bigquery extension

In order to run BigQueries directly in the notebook, you need to load this jupyter extension first


```python
%load_ext google.cloud.bigquery
```

### Check out the dataset

This is dataset represents data from bike sharing om Austin, TX.


```python
# create a reference to our table
table = client.get_table("bigquery-public-data.austin_bikeshare.bikeshare_trips")

# look at five rows from our dataset
client.list_rows(table, max_results=5).to_dataframe()
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
      <th>trip_id</th>
      <th>subscriber_type</th>
      <th>bikeid</th>
      <th>start_time</th>
      <th>start_station_id</th>
      <th>start_station_name</th>
      <th>end_station_id</th>
      <th>end_station_name</th>
      <th>duration_minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9900285908</td>
      <td>Annual Membership (Austin B-cycle)</td>
      <td>400</td>
      <td>2014-10-26 14:12:00+00:00</td>
      <td>2823</td>
      <td>Capital Metro HQ - East 5th at Broadway</td>
      <td>2544</td>
      <td>East 6th &amp; Pedernales St.</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9900289692</td>
      <td>Walk Up</td>
      <td>248</td>
      <td>2015-10-02 21:12:01+00:00</td>
      <td>1006</td>
      <td>Zilker Park West</td>
      <td>1008</td>
      <td>Nueces @ 3rd</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9900285987</td>
      <td>24-Hour Kiosk (Austin B-cycle)</td>
      <td>446</td>
      <td>2014-10-26 15:12:00+00:00</td>
      <td>2712</td>
      <td>Toomey Rd @ South Lamar</td>
      <td>2712</td>
      <td>Toomey Rd @ South Lamar</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9900285989</td>
      <td>24-Hour Kiosk (Austin B-cycle)</td>
      <td>203</td>
      <td>2014-10-26 15:12:00+00:00</td>
      <td>2712</td>
      <td>Toomey Rd @ South Lamar</td>
      <td>2712</td>
      <td>Toomey Rd @ South Lamar</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9900285991</td>
      <td>24-Hour Kiosk (Austin B-cycle)</td>
      <td>101</td>
      <td>2014-10-26 15:12:00+00:00</td>
      <td>2712</td>
      <td>Toomey Rd @ South Lamar</td>
      <td>2712</td>
      <td>Toomey Rd @ South Lamar</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



### 2 - Evaluate the data using a dataframe
Create a dataframe from the query using the BigQuery magic command and have a quick look.

The **%%bigquery** magic command allows you to create a dataframe based on the selection, in this case named df_rides, in order to review the data. 

**Notes:**
* The select is a group by starting station and date and hour if the timestamp column
* The function timestamp_trunc truncates the timestamp column on the hour
* We want to select as training data any records with a date before the first of January 2018


```python
%%bigquery df_rides
select start_station_name, timestamp_trunc(start_time,hour) as start_hour, count(trip_id) as num_rides
  from `bigquery-public-data.austin_bikeshare.bikeshare_trips`
where start_time < '2018-01-01'
  and ( ( start_station_id != end_station_id ) or
        ( start_station_id = end_station_id and duration_minutes > 59 ) )
 group by start_station_name, start_hour
```

Initial look at the query results ...


```python
# check the first fiew rows
df_rides.head()
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
      <th>start_station_name</th>
      <th>start_hour</th>
      <th>num_rides</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Zilker Park West</td>
      <td>2015-10-02 21:00:00+00:00</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Zilker Park West</td>
      <td>2015-10-02 20:00:00+00:00</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zilker Park West</td>
      <td>2015-10-03 21:00:00+00:00</td>
      <td>77</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nueces @ 3rd</td>
      <td>2015-10-03 12:00:00+00:00</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Zilker Park West</td>
      <td>2015-10-03 16:00:00+00:00</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
# how many records are we dealing with here?
df_rides.shape
```




    (292074, 3)



See if there are any issues with the data. Looks like we have no null values and the values of each column has the same data type:


```python
df_rides.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 292074 entries, 0 to 292073
    Data columns (total 3 columns):
    start_station_name    292074 non-null object
    start_hour            292074 non-null datetime64[ns, UTC]
    num_rides             292074 non-null int64
    dtypes: datetime64[ns, UTC](1), int64(1), object(1)
    memory usage: 6.7+ MB


### 3 - Create and train the model
Create the model based on the query. We are using the rows with a start date of before 2018-01-01 for the training data. Later on we will use the rows on or after this date for the test evaluation data.

The "label" (to be predicted) column is the number of rides per hour predicted based on a linear regression model:


```python
%%bigquery

CREATE OR REPLACE MODEL `model_dataset.bike_trips`
OPTIONS(model_type='linear_reg', OPTIMIZE_STRATEGY='batch_gradient_descent') AS
SELECT start_station_name, 
       TIMESTAMP_TRUNC(start_time, HOUR) as start_hour,
       COUNT(trip_id) as label
FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
WHERE start_time < "2018-01-01"
GROUP BY start_station_name, start_hour
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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Look at the iterations of the model


```python
%%bigquery
SELECT
  *
FROM
  ML.TRAINING_INFO(MODEL `model_dataset.bike_trips`)
ORDER BY iteration 
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
      <th>training_run</th>
      <th>iteration</th>
      <th>loss</th>
      <th>eval_loss</th>
      <th>learning_rate</th>
      <th>duration_ms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>4.172457</td>
      <td>3.812986</td>
      <td>0.1</td>
      <td>7083</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>3.944248</td>
      <td>3.603601</td>
      <td>0.2</td>
      <td>12934</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>3.817862</td>
      <td>3.473669</td>
      <td>0.2</td>
      <td>11296</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>3.744511</td>
      <td>3.412568</td>
      <td>0.2</td>
      <td>11264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>3.700585</td>
      <td>3.365890</td>
      <td>0.2</td>
      <td>12100</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>5</td>
      <td>3.673713</td>
      <td>3.347049</td>
      <td>0.2</td>
      <td>13605</td>
    </tr>
  </tbody>
</table>
</div>



### 4 - Model Evaluation
Run the evaluation based on the test data: rows with a start date >= 2018-01-01.

We are especially looking at the R2_score here for the evaluation criteria.


```python
%%bigquery

SELECT *
FROM
ML.EVALUATE(MODEL `model_dataset.bike_trips`, (
SELECT start_station_name, 
       TIMESTAMP_TRUNC(start_time, HOUR) as start_hour,
       COUNT(trip_id) as label
  FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
 WHERE start_time >= "2018-01-01"
 GROUP BY start_station_name, start_hour
))
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
      <th>mean_absolute_error</th>
      <th>mean_squared_error</th>
      <th>mean_squared_log_error</th>
      <th>median_absolute_error</th>
      <th>r2_score</th>
      <th>explained_variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.751198</td>
      <td>13.494581</td>
      <td>0.334176</td>
      <td>0.876819</td>
      <td>-0.087843</td>
      <td>-0.019927</td>
    </tr>
  </tbody>
</table>
</div>



**With a negative R2_score of -0.087843 we are doing worse than using the mean for a prediction so what's up?**

**Note:** for an explaination of r2_score see the following site [http://www.fairlynerdy.com/what-is-r-squared/](http://www.fairlynerdy.com/what-is-r-squared/)

### 5 - Theories for poor performance
* There is no random test train split used
* The year 2018 has a definite increase of rides compared to the other years
* As we are looking at hourly data any bikes returned to the same station within an hour should not be taken into account<br>
  as they will be available with the same hour
* We should be looking at stations that have existed for a while before 2018 and then continued to exist in 2018
* We should be looking at "real" stations and not temporary or administrative ones

Have a look at rides by year, notice the spike in 2018


```python
%%bigquery df_rides_by_year
select extract(year from start_time) as start_year, count(trip_id) as num_rides
  from `bigquery-public-data.austin_bikeshare.bikeshare_trips`
 group by start_year
 order by start_year
```


```python
import matplotlib.pyplot as plt
df_rides_by_year.set_index('start_year',inplace=True)
df_rides_by_year.plot(kind='bar')
plt.show()
```


![png](/images/kagggle-bq-exercise_files/kaggle-bq-exercise_23_0.png)


### 6 - Exercise looking at predictions

A good way to figure out where your model is going wrong is to look closer at a small set of predictions. Use your model to predict the number of rides for the 22nd & Pearl station in 2018. Compare the mean values of predicted vs actual rider


```python
%%bigquery
SELECT AVG(predicted_label) as avg_predicted_trips,
       AVG(label) as avg_actual_trips  
  FROM ML.PREDICT(MODEL `model_dataset.bike_trips`, (
    SELECT start_station_name,
           TIMESTAMP_TRUNC(start_time, HOUR) as start_hour,
           COUNT(trip_id) as label
     FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
    WHERE EXTRACT(YEAR from start_time) = 2018 
      AND start_station_name = '22nd & Pearl'
    GROUP BY start_station_name, start_hour))
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
      <th>avg_predicted_trips</th>
      <th>avg_actual_trips</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.563598</td>
      <td>3.954864</td>
    </tr>
  </tbody>
</table>
</div>



### 7 - Exercise: Average daily rides per station

Either something is wrong with the model or something surprising is happening in the 2018 data.

What could be happening in the data? Write a query to get the average number of riders per station for each year in the dataset and order by the year so you can see the trend. You can use the EXTRACT method to get the day and year from the start time timestamp


```python
%%bigquery df_avg_rides_by_year

WITH stations_by_year AS (
  SELECT start_station_name,
         EXTRACT(year FROM start_time) AS year,
         COUNT(trip_id) as rides
    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
   GROUP BY start_station_name, year
   ORDER BY start_station_name, year
)
SELECT year, AVG(rides) AS avg_rides
  FROM stations_by_year
 GROUP by year
 ORDER by year
```


```python
df_avg_rides_by_year.set_index('year',inplace=True)
df_avg_rides_by_year.plot(kind='bar')
plt.title('Average Rides per Station per Year')
plt.show()
```


![png](/images/kagggle-bq-exercise_files/kaggle-bq-exercise_28_0.png)


### 8 - What do your results tell you?

Given the daily average riders per station over the years, does it make sense that the model is failing?

* Due to the spike in 2018 it does make sense the model is failing
* There are several stations that have been added during 2018 so no history data to predict on
* A suggestion would be to only take stations into account that have existed within a certain time frame like from 2015 to 2019

### 9 - Next Steps
Looking at a [solution](https://www.kaggle.com/evimarp/bigquery-machine-learning-exercise) that Evimar Principal de Soto presented in her kernel which a random selection was chosen to split the train-test set into an 80/20 split, this gave a much better r2_score.

In an attempt to improve the score I have also added a selection of only rides that have a different start and end station or rides that have returned the bicycle to the same station after an hour. Within the hour they would be available again so no need to take into account was the idea


```python
%%bigquery

CREATE OR REPLACE MODEL`model_dataset.bike_trips80`
OPTIONS(model_type='linear_reg',
        OPTIMIZE_STRATEGY='batch_gradient_descent') AS
WITH stations AS
(
    SELECT start_station_name,
           TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour,
           MAX(trip_id) AS trip_id,
           COUNT(trip_id) AS label
      FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
     WHERE start_station_id != end_station_id OR
        ( start_station_id = end_station_id AND duration_minutes > 59 ) 
     GROUP BY start_station_name, start_hour
)
SELECT start_station_name, start_hour, label
  FROM stations
 WHERE MOD(FARM_FINGERPRINT(CAST(trip_id as STRING)), 10)  < 8
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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Evaluate the model based on the new train/test split:


```python
%%bigquery
SELECT
  *
FROM ML.EVALUATE(MODEL `model_dataset.bike_trips80`, (
  WITH stations AS
(
    SELECT start_station_name,
           TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour,
           MAX(trip_id) AS trip_id,
           COUNT(trip_id) AS label
      FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
     WHERE start_station_id != end_station_id OR
         ( start_station_id = end_station_id AND duration_minutes > 59 ) 
     GROUP BY start_station_name, start_hour
)
SELECT start_station_name, start_hour, label
  FROM stations
 WHERE MOD(FARM_FINGERPRINT(CAST(trip_id as STRING)), 10)  >= 8
))
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
      <th>mean_absolute_error</th>
      <th>mean_squared_error</th>
      <th>mean_squared_log_error</th>
      <th>median_absolute_error</th>
      <th>r2_score</th>
      <th>explained_variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.216696</td>
      <td>3.900408</td>
      <td>0.17694</td>
      <td>0.889362</td>
      <td>0.202602</td>
      <td>0.202982</td>
    </tr>
  </tbody>
</table>
</div>



* The mean squared error of 3.900408 has improved compared to the previous model evaluation : 13.494581
* The r2_score is now positive: 0.202602 compared to the previous negative value of -0.087843

### Try again with start stations that have existed from 2016 to 2018

Let's see if this will improve the r2_score by only querying stations that have existed in the period from 2016 to 2018. Note: still doing the random train/test split here.


```python
%%bigquery 

CREATE OR REPLACE MODEL`model_dataset.bike_trips80_2`
OPTIONS(model_type='linear_reg',
        OPTIMIZE_STRATEGY='batch_gradient_descent') AS
WITH conseq_stations AS 
(
  SELECT start_station_name,
         EXTRACT(year FROM start_time) AS year,
         COUNT(trip_id) AS rides
    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
   WHERE EXTRACT(year FROM start_time) BETWEEN 2016 and 2018
   GROUP BY start_station_name, year
   ORDER BY start_station_name, year
),
stations_2016_2018 AS 
(
  SELECT start_station_name
    FROM conseq_stations
   GROUP BY start_station_name
  HAVING COUNT(*) = 3
   ORDER BY start_station_name
),
stations AS
(
    SELECT start_station_name,
           TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour,
           MAX(trip_id) AS trip_id,
           COUNT(trip_id) AS label
      FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
     WHERE EXTRACT(year FROM start_time) BETWEEN 2016 AND 2018
     GROUP BY start_station_name, start_hour
)
SELECT stations.start_station_name, stations.start_hour, stations.label
  FROM stations INNER JOIN stations_2016_2018
    ON stations.start_station_name = stations_2016_2018.start_station_name
 WHERE MOD(FARM_FINGERPRINT(CAST(trip_id as STRING)), 10)  < 8
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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
%%bigquery
SELECT
  *
FROM ML.EVALUATE(MODEL `model_dataset.bike_trips80_2`, (
WITH seq_stations AS 
(
  SELECT start_station_name,
         EXTRACT(year FROM start_time) AS year,
         COUNT(trip_id) AS rides
    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
   WHERE EXTRACT(year FROM start_time) BETWEEN 2016 and 2018
   GROUP BY start_station_name, year
   ORDER BY start_station_name, year
),
stations_2016_2018 AS 
(
  SELECT start_station_name
    FROM seq_stations
   GROUP BY start_station_name
  HAVING COUNT(*) = 3
   ORDER BY start_station_name
),
stations AS
(
    SELECT start_station_name,
           TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour,
           MAX(trip_id) AS trip_id,
           COUNT(trip_id) AS label
      FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
     GROUP BY start_station_name, start_hour
)
SELECT stations.start_station_name, stations.start_hour, stations.label
  FROM stations INNER JOIN stations_2016_2018
    ON stations.start_station_name = stations_2016_2018.start_station_name
 WHERE MOD(FARM_FINGERPRINT(CAST(trip_id as STRING)), 10)  >= 8
))
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
      <th>mean_absolute_error</th>
      <th>mean_squared_error</th>
      <th>mean_squared_log_error</th>
      <th>median_absolute_error</th>
      <th>r2_score</th>
      <th>explained_variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.196344</td>
      <td>3.491311</td>
      <td>0.174068</td>
      <td>0.914008</td>
      <td>0.123441</td>
      <td>0.123603</td>
    </tr>
  </tbody>
</table>
</div>



* The mean squared error of 3.491311 has slightly improved compared to the previous model evaluation : 3.900408
* The r2_score has not improved: 0.10704 compared to the previous value 0.202602

It looks like this hasn't improved the results
