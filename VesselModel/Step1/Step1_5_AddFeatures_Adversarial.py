from matplotlib.patches import Polygon
import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# df = pd.read_csv("~/BCFerryData/feature1.csv")
df = pd.read_csv("data/Features/feature1_tmp2_2.csv")


plt.scatter(df.LONGITUDE, df.LATITUDE, c = df.MODE, s=1)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()


# find detour trips

def determine_bound(x1,y1,x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y1 - m * x1
    return m,b

def points_outOfBound(row,lowerM, lowerB, upperM, upperB):
    x = row['LONGITUDE']
    y = row['LATITUDE']
    if ( (x > -123.9) & (y < 49.28)):
        y_line = lowerM * x + lowerB
        return y < y_line
    if ((x < -123.35) & (y > 49.28)):
        y_line = upperM * x + upperB
        return y > y_line
    return False
    
    
lowerBound_m, lowerBound_b = determine_bound(-123.9014, 49.2038, -123.7031, 49.2317)
# upperBound_m, upperBound_b = determine_bound(-123.6169, 49.3274, -123.3247, 49.4316)
upperBound_m, upperBound_b = determine_bound(-123.7343, 49.2769, -123.5345, 49.3168)

df["specialPoint"] = df.apply(points_outOfBound, axis=1, args=(lowerBound_m, lowerBound_b, upperBound_m, upperBound_b))

# samples = df[df["specialPoint"] == True].trip_id.unique()

def detourTrips(df, detour_thresh = 5):
    special = []
    for trip_id in list(df.trip_id.unique()):
        tmp_df = df[df.trip_id == trip_id]
        specialCount = tmp_df['specialPoint'].sum()
        if (specialCount > detour_thresh):
            special.append(trip_id)
    return special


detourTrip = detourTrips(df)
df['detourTrip'] = df['trip_id'].apply(lambda trip_id: 1 if trip_id in detourTrip else 0)

samples = df[df["detourTrip"] == True].trip_id.unique()
plt.scatter(df[df.trip_id.isin(samples)].LONGITUDE, df[df.trip_id.isin(samples)].LATITUDE, c = df[df.trip_id.isin(samples)].STW, s=1)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

df = df.drop(["specialPoint","detourTrip"], axis = 1)
plt.scatter(df[df.trip_id==1049].LONGITUDE,df[df.trip_id==1049].LATITUDE, c = df[df.trip_id==1049].STW, s=1)

def mode_change_inbetween(df, speed_thresh=10):
    trip_lists = []
    for trip_id in list(df.trip_id.unique()):
        tmp_df = df[df.trip_id==trip_id]
        tmp_df = tmp_df[(tmp_df.LONGITUDE>-123.84) & (tmp_df.LONGITUDE<-123.4)]
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
    adversarial = list(set(reduce_speed + reroute + detourTrip))
    return adversarial, reroute, reduce_speed

adversarial, reroute, reduce_speed = get_adversarial(df)
df["adversarial"] = df["trip_id"].apply(lambda x: 1 if x in adversarial else 0)

len(adversarial) #361 trips
# len(df.trip_id.unique()) #3129 trips
samples = reduce_speed
# plt.scatter(df[df.trip_id.isin(samples)].LONGITUDE, df[df.trip_id.isin(samples)].LATITUDE, c = df[df.trip_id.isin(samples)].STW,s=1)
plt.scatter(df[df.trip_id.isin(samples)].LONGITUDE, df[df.trip_id.isin(samples)].LATITUDE, c = df[df.trip_id.isin(samples)].MODE,s=1)

samples = adversarial
plt.scatter(df[df.trip_id.isin(samples)].LONGITUDE, df[df.trip_id.isin(samples)].LATITUDE, c = df[df.trip_id.isin(samples)].STW, s=1)

# len(detourTrip)
# len(adversarial)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

plt.scatter(df[~df.trip_id.isin(samples)].LONGITUDE, df[~df.trip_id.isin(samples)].LATITUDE, c = df[~(df.trip_id.isin(samples))].MODE,s=1)

#1143,1591,2048,2630
# print(reroute) #[135, 1032, 1039, 1075, 1101, 1143, 1312, 1591, 1825, 1904, 2048, 2142, 2208, 2399, 2630, 3001, 3216, 3245, 3351, 3373, 3671, 3742]
# idx = 21
# print(reroute[idx])
# id = reroute[idx]
# plt.scatter(df[df.trip_id==id].LONGITUDE, df[df.trip_id==id].LATITUDE, s=1)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.show()
# df = df[~df.trip_id.isin(reroute)]


# df.to_csv("~/BCFerryData/feature1.csv", index=False)
df.to_csv("data/Features/feature1.csv", index=False)
# df.to_csv("data/Features/feature1.csv", index=False)




def plot_location(samples, column, s=1):
    plt.scatter(samples.LONGITUDE, samples.LATITUDE, c = samples[column], s=1)
    plt.colorbar(fraction=0.046, pad=0.04).ax.tick_params(labelsize=7, rotation=90)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7,rotation=90)
    plt.title("location vs. {}".format(column), fontsize=7)

sample_size=20
sample_trips = df.trip_id.sample(sample_size)
samples = df[df.trip_id.isin(sample_trips)]

fig = plt.figure(figsize=(9,6))
fig.add_subplot(2,2,1)
plot_location(samples, "PITCH_1", s=1)
fig.add_subplot(222)
plot_location(samples, "SPEED_1", s=1)
fig.add_subplot(223)
plot_location(samples, "PITCH_2", s=1)
fig.add_subplot(224)
plot_location(samples, "SPEED_2", s=1)
fig.show()

fig = plt.figure(figsize=(9,6))
fig.add_subplot(2,2,1)
plot_location(samples, "ENGINE_1_FUEL_CONSUMPTION", s=1)
fig.add_subplot(222)
plot_location(samples, "SPEED_1", s=1)
fig.add_subplot(223)
plot_location(samples, "ENGINE_2_FUEL_CONSUMPTION", s=1)
fig.add_subplot(224)
plot_location(samples, "SPEED_2", s=1)
fig.show()



fig = plt.figure(figsize=(9,6))
fig.add_subplot(2,2,1)
plot_location(samples, "countDown", s=1)
fig.add_subplot(222)
plot_location(samples, "ScheduleType", s=1)
# fig.add_subplot(223)
# plot_location(samples, "", s=1)
# fig.add_subplot(224)
# plot_location(samples, "", s=1)
fig.show()


df.columns