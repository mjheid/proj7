import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter
import sys

# load data which is a list of countries/locatons in csv format
df = pd.read_csv("data/countries.txt", header=None).rename(columns={0: "country1"})

# get geolocator
geolocator = Nominatim(user_agent="dist-btw-ctrs")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# get positions of countries
df["Cor1"] = df["country1"].apply(geocode)

# check if all countries got found, if not exit
if df["Cor1"].isnull().sum() > 0:
    null_series = df["Cor1"].isnull()
    for index, row in df.iterrows():
        if null_series[index]:
            country_not_found = df["country1"][index]
            print(f"WARNING: unable to find a location for  {country_not_found}!")
    print("Fix country names and run again!")
    sys.exit()


# create DataFrame in which we will calculate distances
data = pd.DataFrame(columns = ['country1', 'country2', 'distance', 'Cor1', 'Cor2'])

# create all pairs of countries and their positions
start_pos = 0
step_size = df.shape[0]
end_pos = step_size
for i in list(range(df.shape[0])):
    data = pd.concat([data,df], ignore_index=True)
    data["country2"][start_pos:end_pos] = df["country1"][i]
    for j in list(range(start_pos, end_pos)):
        data["Cor2"][j] = df["Cor1"][i]
    start_pos = end_pos
    end_pos += step_size


# Create columns with lat and lon data ofthe Cor1 and Cor2 columns 
data["lat1"] = data["Cor1"].apply(lambda x: x.latitude if x != None else None)
data["lon1"] = data["Cor1"].apply(lambda x: x.longitude if x != None else None)
data["lat2"] = data["Cor2"].apply(lambda x: x.latitude if x != None else None)
data["lon2"] = data["Cor2"].apply(lambda x: x.longitude if x != None else None)


def distance(data):
    """
    arg: data: DataFrame with lat1,lat2,lon1,lon2
            data columns.
    describtion: Calculate distance between each pair
                of distances
    """ 

    # get positions of countries
    country1 = list(zip(data.lat1, data.lon1)) 
    country2 = list(zip(data.lat2, data.lon2))

    # calculate dist between each country pair
    x = [country1, country2]
    dist = []
    for i in list(range(len(country1))):
        dist.append(geodesic(x[0][i], x[1][i]).miles)
    
    return dist

data["distance"] = distance(data)

# Set distances of zero to 100 -> makes it less likely that people stay in country
tmp=data[data["distance"]==0]
tmp["distance"]=100.0
data[data["distance"]==0] = tmp

# convert distances into inverse liklyhoods
start_pos = 0
step_size = df.shape[0]
end_pos = step_size
data["p_dist"] = 0.0
for i in list(range(step_size)):
    df = data[start_pos:end_pos]
    df["inf_dist"] = df["distance"].apply(lambda x: 1/x)
    inf_dist_sum = df["inf_dist"].sum()
    data["p_dist"][start_pos:end_pos] = df["inf_dist"].apply(lambda x: x/inf_dist_sum)
    start_pos = end_pos
    end_pos += step_size


# write to csv
data.to_csv("data/p_distances.csv", mode="w", columns=["country1", "country2", "p_dist"], index=False)
