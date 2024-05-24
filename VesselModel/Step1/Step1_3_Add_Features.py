import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt

file_path = "~/BCFerryData/df_naive_impute.csv"
# df = pd.read_csv(file_path, skiprows=[1])
df = pd.read_csv(file_path)

df.columns, len(df.columns)

# naive mode definition
df["POWER_1"] = df["POWER_1"].astype(float)
df["POWER_2"] = df["POWER_2"].astype(float)
df["STW"] = df["STW"].astype(float)
df["STW_diff"] = abs(df["STW"] - df["STW"].shift(periods=-1, fill_value=100))

# MODE for operation mode: 1 for mode1, 2 for mode2, 0 for unknown(power with nan)
def naive_operation_mode(row):
    if pd.isna(row['POWER_1']) or (pd.isna(row["POWER_2"])):
        return 0
    elif (~((row["POWER_1"]<=800) ^ (row["POWER_2"]<=800))) | (row["STW"]<=16) | (row["STW_diff"]>=1):
        return 2
    else:
        return 1

df["MODE"] = df[["POWER_1", "POWER_2", "STW", "STW_diff"]].apply(naive_operation_mode, axis=1)
df.drop("STW_diff", axis=1, inplace=True)

def correct_mode(row):
    if (row["MODE"]==1):
        if row["trip_id"]==0:
            return 2
    return row["MODE"]

def clean_mode(row):
    if row["MODE_up"] == row["MODE_down"]:
        if row["MODE"]!=row["MODE_up"]:
            return row["MODE_up"]
    return row["MODE"]


df["MODE"] = df[["MODE", "STW", "DEPTH", "trip_id"]].apply(correct_mode,axis=1)
#df.head(120)
df["MODE_up"] = df["MODE"].shift(periods=-1)
df["MODE_down"] = df["MODE"].shift(periods=1)
df["MODE"] = df[["MODE", "MODE_up", "MODE_down"]].apply(clean_mode,axis=1)
df.drop(["MODE_up", "MODE_down"],axis=1, inplace=True)

samples = df.trip_id.sample(100)
plt.scatter(df[df.trip_id.isin(samples)].LONGITUDE, df[df.trip_id.isin(samples)].LATITUDE, c = df[df.trip_id.isin(samples)].MODE, s=1)

df.groupby("MODE").count()

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

# from the Dati variable, get the corresponding season code, hour, and day of the week
def get_dt_info(minute, starting_dt):
    dt = starting_dt + datetime.timedelta(minutes=minute) # days, seconds, then other fields.
    month = dt.month
    season = get_season(month)
    hour = dt.hour
    weekday = dt.weekday()
    return dt, season, hour, weekday

# get season code from corresponding month:
# spring: 0, summer: 1, fall: 2, winter: 3
def get_season(month):
    if month <= 3:
        return 3
    elif month <=6:
        return 0
    elif month <=9:
        return 1
    else:
        return 2

# def get_season(month):
#     if month <= 3:
#         return 0
#     elif month <=6:
#         return 1
#     elif month <=9:
#         return 2
#     else:
#         return 3
    

# convert Dati to python datetime format
df["datetime"] = pd.to_datetime(df.Dati, format='%y%m%d_%H%M%S')
df["month"] = df["datetime"].dt.month
df.drop(['month'], axis=1, inplace=True)
df["season"] = df['datetime'].apply(lambda x: get_season(x.month))
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday
# df.dropna(subset = ["Time"], inplace=True)
# starting_dt = pd.to_datetime(df.iloc[0].Dati, format='%y%m%d_%H%M%S')

# print(starting_dt)
# starting_dt.weekday()
#print(df.iloc[0].Dati)

#df["datetime"],df["season"], df["hour"], df["weekday"] = zip(*df["Time"].apply(get_dt_info, starting_dt = starting_dt))
# df["datetime"] = pd.to_datetime(df.Dati, format='%y%m%d_%H%M%S')
df[["datetime", "season","hour", "weekday"]]
#df[800:900][["Dati","datetime", "season","hour", "weekday",'trip_id']]


df.trip_id.unique()
df[df.trip_id==5][["datetime", "weekday"]]

# current: difference between STW and SOG
df["current"] = df["STW"] - df["SOG"]
# df.current


def get_direction():
    trips = list(df.trip_id.unique())
    # trip 0 (parking) has direction 0
    direction_dict = {}
    direction_dict[0] = "parking"
    for trip in trips:
        tmp_df = df[df.trip_id==trip].reset_index()
        if (tmp_df.iloc[-1].LONGITUDE > tmp_df.iloc[0].LONGITUDE) & (tmp_df.iloc[-1].LATITUDE > tmp_df.iloc[0].LATITUDE):
            direction_dict[trip]="N-H"
        elif (tmp_df.iloc[-1].LONGITUDE < tmp_df.iloc[0].LONGITUDE) & (tmp_df.iloc[-1].LATITUDE < tmp_df.iloc[0].LATITUDE):
            direction_dict[trip]="H-N"
        else:
            print("error occurs when classifying trip {}".format(trip))
    return direction_dict

# df.to_csv('~/BCFerryData/tmp1.csv', index=False)
# df = pd.read_csv('~/BCFerryData/tmp1.csv')

direcs_dict = get_direction() # error at 1352, 1352 only includes one row of data

#print(df.groupby(df.trip_id).count().Dati[df.groupby(df.trip_id).count().Dati<90])

# plt.scatter(df[df.trip_id==1352].LONGITUDE, df[df.trip_id==1352].LATITUDE, s=2, c=df[df.trip_id==1352].Time, cmap="BrBG")
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.show()
df = df[df.trip_id!=1352]

df["direction"] = df["trip_id"].apply(lambda x: direcs_dict[x])
df.groupby("direction").count()

# weather related features
# temperature, humidity, pressure, precipitation, rain, snowfall
# weathercode: uses WMO weather codes


weather = pd.read_csv(os.getcwd() + "/data/weather.csv")
weather.columns = ["time", "pressure", "rain", "snowfall", "weathercode"]
weather["time"] = pd.to_datetime(weather["time"], format='%Y-%m-%dT%H:%M')
weather["day"] = weather.time.apply(lambda x: x.date())
weather["hour"] = weather.time.apply(lambda x: x.hour)

df["day"] = df.datetime.apply(lambda x: x.date())
#df["day"] = df["datetime"].dt.day
df["hour"] = df.datetime.apply(lambda x: x.hour)
df = pd.merge(df, weather, on=["day", "hour"], how="left").drop(["day", "hour", "time"], axis=1)

holidays = []
holidays.append(datetime.datetime(2019, 9, 2))
holidays.append(datetime.datetime(2019, 10, 14))
holidays.append(datetime.datetime(2019, 11, 11))
holidays.append(datetime.datetime(2019, 12, 25))
holidays.append(datetime.datetime(2020, 1, 1))
holidays.append(datetime.datetime(2020, 2, 17))
holidays.append(datetime.datetime(2020, 4, 10))
holidays.append(datetime.datetime(2020, 5, 18))
holidays.append(datetime.datetime(2020, 7, 1))
holidays.append(datetime.datetime(2020, 8, 3))
holidays.append(datetime.datetime(2020, 9, 7))
holidays.append(datetime.datetime(2020, 10, 12))
holidays.append(datetime.datetime(2020, 11, 11))
holidays.append(datetime.datetime(2020, 12, 25))
holidays.append(datetime.datetime(2021, 1, 1))
holidays.append(datetime.datetime(2021, 2, 15))
holidays.append(datetime.datetime(2021, 4, 2))
holidays.append(datetime.datetime(2021, 5, 24))
holidays.append(datetime.datetime(2021, 7, 1))
holidays.append(datetime.datetime(2021, 8, 2))
holidays.append(datetime.datetime(2021, 9, 6))
is_holiday = [1 for i in range(len(holidays))]
holidays = pd.DataFrame({"date":holidays, "holiday":is_holiday})
holidays["date"] = holidays["date"].apply(lambda x: x.date())

df["date"] = df.datetime.apply(lambda x: x.date())
df = pd.merge(df, holidays, on="date", how="left")
df["holiday"] = df["holiday"].fillna(0)
df["is_weekday"] = (df["weekday"]>0) & (df["weekday"]<=5) & (~(df["holiday"]==1))
df = df.drop(["date","holiday"], axis=1)

def get_wind_direction(angle):
    angle = np.abs(angle)
    if angle <= 60:
        return 0
    elif angle <= 120:
        return 1
    elif angle <= 180:
        return 2
    else:
        return np.nan
    
df["wind_force"] = df["WIND_SPEED_TRUE"] ** 2
df["wind_direc"] = df["WIND_ANGLE"].apply(get_wind_direction)

df["effective_wind_factor"] = df[["HEADING", "WIND_ANGLE"]].apply(lambda x: np.cos((x["HEADING"]-x["WIND_ANGLE"])*np.pi/180), axis=1)
df["effective_wind"] = df["WIND_SPEED"] * df["effective_wind_factor"]

df["resist_ratio1"] = df["THRUST_1"]/(df["TORQUE_1"]*df["SPEED_1"]+1e-6)
df["resist_ratio2"] = df["THRUST_2"]/(df["TORQUE_2"]*df["SPEED_2"]+1e-6)

def mode_change_inbetween(df, speed_thresh=10):
    trip_lists = []
    for trip_id in list(df.trip_id.unique()):
        tmp_df = df[df.trip_id==trip_id]
        tmp_df = tmp_df[(tmp_df.LONGITUDE>-123.8) & (tmp_df.LONGITUDE<-123.4)]
        tmp_df["MODE_diff"] = abs(tmp_df["MODE"]-tmp_df["MODE"].shift(periods=-1))
        mode_change = tmp_df["MODE_diff"].sum()
        if mode_change > 0:
            trip_lists.append(trip_id)
        # if (tmp_df.mode<speed_thresh).sum()>0:
            # trip_lists.append(trip_id)
    return trip_lists
        
def travel_distance(df, thresh = 1.3):
    trip_distance = {}
    for trip_id in list(df.trip_id.unique()):
        tmp_df = df[df.trip_id==trip_id]
        tmp_df["LONGITUDE_up"] = tmp_df["LONGITUDE"].shift(periods=-1, fill_value=tmp_df.iloc[-1].LONGITUDE)
        tmp_df["LATITUDE_up"] = tmp_df["LATITUDE"].shift(periods=-1, fill_value=tmp_df.iloc[-1].LATITUDE)
        tmp_df["distance"] = ((tmp_df["LONGITUDE_up"]-tmp_df["LONGITUDE"])**2 + (tmp_df["LATITUDE_up"]-tmp_df["LATITUDE"])**2)**0.5
        trip_distance[trip_id] = tmp_df["distance"].sum()
    mean = np.mean(list(trip_distance.values()))
    std = np.std(list(trip_distance.values()))
    threshold = mean + std * thresh
    rerout_list = []
    for _, (trip, dist) in enumerate(trip_distance.items()):
        if dist > threshold:
            rerout_list.append(trip)
    return (rerout_list)

def get_adversarial(df):
    reduce_speed = mode_change_inbetween(df)
    reroute = travel_distance(df)
    adversarial = list(set(reduce_speed+reroute))
    return adversarial, reroute, reduce_speed

adversarial, reroute, reduce_speed = get_adversarial(df)
df["adversarial"] = df["trip_id"].apply(lambda x: 1 if x in adversarial else 0)

len(adversarial)

samples = reroute
plt.scatter(df[df.trip_id.isin(samples)].LONGITUDE, df[df.trip_id.isin(samples)].LATITUDE, s=1)

samples = reduce_speed
plt.scatter(df[df.trip_id.isin(samples)].LONGITUDE, df[df.trip_id.isin(samples)].LATITUDE, c = df[df.trip_id.isin(samples)].STW, s=1)


plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

df.to_csv("~/BCFerryData/feature1.csv", index=False)

len(df[pd.isna(df)].trip_id)
# df
# df.columns


#pd.set_option("display.max_rows",10)