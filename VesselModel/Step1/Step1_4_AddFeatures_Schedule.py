import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pickle
import matplotlib.pyplot as plt
import os

# file_path = "~/BCFerryData/df_naive_impute.csv"
file_path = "data/Features/feature_tmp1.csv"
# df = pd.read_csv(file_path, skiprows=[1])
df = pd.read_csv(file_path)


# # from the Dati variable, get the corresponding season code, hour, and day of the week
# def get_dt_info(minute, starting_dt):
#     dt = starting_dt + datetime.timedelta(minutes=minute) # days, seconds, then other fields.
#     month = dt.month
#     season = get_season(month)
#     hour = dt.hour
#     weekday = dt.weekday()
#     return dt, season, hour, weekday



# convert Dati to python datetime format
df["datetime"] = pd.to_datetime(df.Dati, format='%y%m%d_%H%M%S')
df["month"] = df["datetime"].dt.month
# df["day"] = df["datetime"].dt.day
# df.drop(['month'], axis=1, inplace=True)
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday

# df.head(20)

# Finding Schedules
# round down datetime to nearest 5 minutes
df["Rounded_DT"] = df['datetime'].dt.floor('5T')

df["Rounded_Time"] = df["Rounded_DT"].dt.time
df.head(20)

departureTime = df.groupby('trip_id').head(1).reset_index(drop=True)
arrivalTime = df.groupby('trip_id').tail(1).reset_index(drop=True)

tripSchedule = departureTime[["trip_id","direction"]]
tripSchedule["date"] = departureTime["datetime"].dt.date
tripSchedule["year"] = departureTime["datetime"].dt.year
tripSchedule["month"] = departureTime["datetime"].dt.month
tripSchedule["departure"] = departureTime["datetime"]
tripSchedule["arrival"] = arrivalTime["datetime"]
tripSchedule["duration"] = (tripSchedule["arrival"] - tripSchedule["departure"]).dt.total_seconds() /60


tripSchedule["RoundedDT"] = tripSchedule["departure"].dt.floor('5T')
tripSchedule["RoundedTime"] = tripSchedule["departure"].dt.floor('5T').dt.time
# tripSchedule

# tripSchedule.to_csv("data/Features/tripSchedule.csv", index=False)

NH_trips = tripSchedule[tripSchedule["direction"] == 'N-H']
HN_trips = tripSchedule[tripSchedule["direction"] == 'H-N']





# NH_trips.groupby(['year','month'])["RoundedTime"].value_counts().reset_index(name = 'count').head(50)

# NH_trips["RoundedTime"].value_counts()


# NH_possibleSchedule = NH_trips["RoundedTime"].value_counts()[NH_trips["RoundedTime"].value_counts()]
# NH_possibleSchedule[NH_trips.head(1)["RoundedTime"]]

def possibleSchedule(df, thresh=1):
    possibleSchedule = df.groupby(['year','month'])["RoundedTime"].value_counts().reset_index(name = 'count')
    # print(possibleSchedule.head(40))
    possibleSchedule = possibleSchedule[possibleSchedule['count'] > thresh]
    return possibleSchedule


def assignSchedule(row, possibleSchedule):
    matches = possibleSchedule[(possibleSchedule['year'] == row['year']) &
                                (possibleSchedule['month'] == row['month']) &
                                (possibleSchedule['RoundedTime'] == row['RoundedTime'])]
    if not matches.empty:
        return row['RoundedDT']
    else:
        return None

# def assignSchedule_lessFreq(row, possibleSchedule):

def roundDown(dt, minute = 5):
    dt = dt - timedelta(minutes=minute)
    return dt.time()
    

# def assignNaNSchedule(df, possibleSchedule):
#      df.loc[df.ScheduleTime.isna(), 'RoundedTime'] = df[df.ScheduleTime.isna()]['RoundedDT'].apply(roundDown)
#      df = df.apply(assignSchedule, axis = 1, possibleSchedule = possibleSchedule)
    



# Setup for N-H trips
NH_possibleSchedule = possibleSchedule(NH_trips)
# Set up Schedule if there is 2 or more trips depature at same time in a month
NH_trips["ScheduleTime"] = NH_trips.apply(assignSchedule, axis = 1, possibleSchedule = NH_possibleSchedule)
# Set up Schdule if there is only 1 trip departure at the time
# Round down 5 more minutes and find the schedule
# NH_trips["ScheduleTime"] = NH_trips.apply(assignNaNSchedule, axis = 1, possibleSchedule = NH_possibleSchedule)

NH_trips.loc[NH_trips.ScheduleTime.isna(), 'RoundedTime'] = NH_trips[NH_trips.ScheduleTime.isna()]['RoundedDT'].apply(roundDown)
NH_trips["ScheduleTime"] = NH_trips.apply(assignSchedule, axis = 1, possibleSchedule = NH_possibleSchedule)
# Set up Schedule time as rounded departure time
NH_trips_NaN = NH_trips[NH_trips.ScheduleTime.isna()]
NH_trips_NaN['ScheduleTime'] = NH_trips_NaN['RoundedDT']
NH_trips.update(NH_trips_NaN)
# NH_trips.head(30)

# len(NH_trips[NH_trips["ScheduleTime"].notna()])
# len(NH_trips[NH_trips["ScheduleTime"].isna()])
# len(NH_trips)

# Setup for H-N trips
HN_possibleSchedule = possibleSchedule(HN_trips)
# Set up Schedule if there is 2 or more trips depature at same time in a month
HN_trips["ScheduleTime"] = HN_trips.apply(assignSchedule, axis = 1, possibleSchedule = HN_possibleSchedule)
# Set up Schdule if there is only 1 trip departure at the time
# Round down 5 more minutes and find the schedule
# HN_trips["ScheduleTime"] = HN_trips.apply(assignNaNSchedule, axis = 1, possibleSchedule = HN_possibleSchedule)

HN_trips.loc[HN_trips.ScheduleTime.isna(), 'RoundedTime'] = HN_trips[HN_trips.ScheduleTime.isna()]['RoundedDT'].apply(roundDown)
HN_trips["ScheduleTime"] = HN_trips.apply(assignSchedule, axis = 1, possibleSchedule = HN_possibleSchedule)
# Set up Schedule time as rounded departure time
HN_trips_NaN = HN_trips[HN_trips.ScheduleTime.isna()]
HN_trips_NaN['ScheduleTime'] = HN_trips_NaN['RoundedDT']
HN_trips.update(HN_trips_NaN)
# len(NH_trips)
# len(HN_trips[HN_trips["ScheduleTime"].isna()])

# NH_trips[['trip_id','date','RoundedDT','RoundedTime']]
# NH_trips[['trip_id','date','RoundedDT','RoundedTime','ScheduleTime']]

# merge SchduleTime to df
df = df.merge(NH_trips[['trip_id', 'ScheduleTime']], on='trip_id', how='left')
df = df.merge(HN_trips[['trip_id', 'ScheduleTime']], on='trip_id', how='left')
df['Schedule'] = df['ScheduleTime_x'].combine_first(df['ScheduleTime_y'])
# df['Schedule'].isna().sum()

#set schedule type
# 0 if day trip between 6am - 11pm, else night trip
start_time = datetime.datetime.strptime('06:00:00', '%H:%M:%S').time()
end_time = datetime.datetime.strptime('23:00:00', '%H:%M:%S').time()
def set_schedule_time(schedule_time):
    if start_time <= schedule_time.time() <= end_time:
        return 0
    else:
        return 1
    
df['ScheduleType'] = df['Schedule'].apply(set_schedule_time)

# set count down
duration = datetime.timedelta(hours=1, minutes=40)
df["EST"] = df["Schedule"] + duration
df["countDown"] = (df["EST"] - df["datetime"]).apply(lambda x: x.total_seconds()//60)
df = df.drop(columns=['ScheduleTime_x','ScheduleTime_y','Rounded_DT','Rounded_Time','EST'])



# test
# df[df["trip_id"]==3766].tail(10)[['datetime','Schedule','countDown','ScheduleType']]



# get season code from corresponding month:
# spring: 0011, summer: 0110, fall: 1100, winter: 1001
def get_season(month):
    if month <= 4 & month >=2: #spring month 2-4
        return '0011'
    elif month <=8 & month >=5: #summer month 5-8
        return '0110'
    elif month <=10 & month >=9: # fall month 9-10
        return '1100'
    else: # winter month 11-1
        return '1001' 

df["season"] = df['datetime'].apply(lambda x: get_season(x.month))
# df[["datetime", "season","hour", "weekday"]]

# df.trip_id.unique()
# df[df.trip_id==5][["datetime", "weekday"]]



# current: difference between STW and SOG
df["current"] = df["STW"] - df["SOG"]
# df.current


# weather related features
# temperature, humidity, pressure, precipitation, rain, snowfall
# weathercode: uses WMO weather codes


weather = pd.read_csv(os.getcwd() + "/data/Features/weather.csv")
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
# df.columns
df = df.drop(["date"], axis=1)



# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.show()

# df.to_csv("~/BCFerryData/feature1_tmp2.csv", index=False)
df.to_csv("data/Features/feature1_tmp2_2.csv", index=False)

# len(df[pd.isna(df)].trip_id) # return 324164
# df
# df.columns

