import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


#file_data = '~/BCFerryData/queenCsvOut.csv'
file_data = "data/Features/queenCsvOut.csv"

if not os.path.exists(file_data):
    print("File not found. PLease put the queenCsvOut in the location: data/queenCsvOut.csv")
    exit()
# df = pd.read_csv('~/BCFerryData/queenCsvOut.csv', skiprows=[1])
df = pd.read_csv(file_data, skiprows=[1])
df


# plot_df = df[df.LATITUDE>48.9]
# plt.scatter(plot_df.LONGITUDE, plot_df.LATITUDE, s=1)
# plot_df = df[df.LATITUDE>48.9]
# tmp = plot_df[(plot_df["LATITUDE"]<49.1) & (plot_df["LONGITUDE"]> -123.4)]
# plt.scatter(tmp.LONGITUDE, tmp.LATITUDE, s=2)


# remove trips that are nether N-H nor H-N trips
df[pd.isna(df.LONGITUDE) & (df.THRUST_1==0) &(df.THRUST_2==0)].sum()[df[pd.isna(df.LONGITUDE) & (df.THRUST_1==0) &(df.THRUST_2==0)].sum()==0]

df = df[~(pd.isna(df.LONGITUDE) & (df.THRUST_1==0) &(df.THRUST_2==0))]
df
# df[(df.SOG == 0) & (df.LONGITUDE.isna()) & (df.LATITUDE.isna())]
# df = df[~((df.SOG == 0) & (df.LONGITUDE.isna()) & (df.LATITUDE.isna()))]


# trip ID

# divide the data into trips, each with a unique trip id. The records have trip id 0 when the vessel is parking at bay
# Horseshoe Bay
# 49.3771, -123.2715
H_lat = 49.3771
H_long = -123.2715

# Nanaimo 
# 49.1936, -123.9554
N_lat = 49.1936
N_long = -123.9554

# Give each trip an ID (from H to N or N to H is counted as a complete trip)
# thresh: threshold for the area of the bay
def number_trip(bay_thresh = 1e-6, speed_thres=1):
    trip = np.zeros(df.shape[0])
    trip_id = 1
    trip[0] = trip_id
    prev_at_bay = True
    flag = True
    for i in range(1, len(df)):
        if (i % 50000)==0:
            print(i, len(df))
        H_dist = (df.iloc[i].LONGITUDE - H_long)**2 + (df.iloc[i].LATITUDE - H_lat)**2
        N_dist = (df.iloc[i].LONGITUDE - N_long)**2 + (df.iloc[i].LATITUDE - N_lat)**2
        # decide if the vessel in near the bay
        at_bay = (H_dist < bay_thresh) | (N_dist < bay_thresh)
        if (at_bay):
            # just enter the bay area
            # use flag to check if a new trip is counted
            if (prev_at_bay==False):
                flag = False
            # slows down, means the vessel is likely to be arrived
            # if hasn't generate a new trip id, do so
            if (flag==False) & (df.iloc[i].SOG <= speed_thres):
                trip_id += 1
                flag = True
            # if the vessel speed is very low near the bay area
            # parking at the bay, set trip id to 0
            if (df.iloc[i].SOG <= speed_thres):
                trip[i] = 0
            else:
                trip[i] = trip_id
        else:
            # if leave the bay, but new trip id hasn't been assigned
            # assgin new trip id
            if flag==False:
                trip_id += 1
                flag=True
            trip[i] = trip_id
        prev_at_bay = at_bay
    return trip

df.dropna(axis=0, thresh=35, inplace=True)
df["trip_id"] = number_trip().astype(int)

print(df.trip_id.min(), df.trip_id.max()) ## total 4093 trips

print(df.groupby(df.trip_id).count().head(20).Dati)
#df[['Dati','Time','LONGITUDE','LATITUDE','SOG','THRUST_1','THRUST_2','trip_id']].head(160)
#df[['Dati','Time','LONGITUDE','LATITUDE','SOG','trip_id']].tail(100)


# df.to_csv("~/BCFerryData/queenCsvOut_withID.csv", index=False)
# df.to_csv("data/Features/queenCsvOut_withID.csv", index=False)
# df = pd.read_csv("~/BCFerryData/queenCsvOut_withID.csv")
# df = pd.read_csv("data/Features/queenCsvOut_withID.csv")


# print(df.groupby(df.trip_id).count()) 

# These are trips with extremely off locations, drop these trips
off_locations = list(df[df.LATITUDE<49.1].trip_id.unique())
# off_locations #2433, 2805, 3115, 3484, 3493, 3707
df = df[~df.trip_id.isin(off_locations)]

#id = 3115 #3484
#id = 3707
#plt.scatter(df[df.trip_id==id].LONGITUDE, df[df.trip_id==id].LATITUDE, s=1)
# plt.scatter(df[df.trip_id==4].LONGITUDE, df[df.trip_id==4].LATITUDE, s=1)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.show()
#df[df.trip_id==3707][['Dati','Time','LONGITUDE','LATITUDE','SOG','trip_id']]



# Clean up grouping
# The grouping is not very accurate, for example, there are incomplete trips due to missingdata
#  and abnormal trips that has the same starting and end point. This could be caused by two or 
# more trips classfied into one (long_trips), or a short movement misclassfied into a single trip (short_trips). 

# Typically, a trip from N-H or H-N will take around 100-107 mins (with corresponding lines of data), the median is at 104.
# Apperently any trip too long or too short is questionable.

print("\n quantile of dati grouped by trips \n")
print(df.groupby(df.trip_id).count().Dati.quantile(0.25), df.groupby(df.trip_id).count().Dati.quantile(0.5), df.groupby(df.trip_id).count().Dati.quantile(0.75))



def get_discontinuous_trip(df, time_thres=10):
    discontinous = {}
    trips = list(df.trip_id.unique())
    trips.remove(0)
    for i in trips:
        tmp_df = df[df.trip_id==i].reset_index()
        fill_in = tmp_df.Time.max()
        tmp_df["time_up"] = tmp_df["Time"].shift(periods=-1, fill_value=fill_in)
        tmp_df["time_diff"] = tmp_df["time_up"] - tmp_df["Time"]
        # tmp_df["time_diff"] = tmp_df["Time"].diff(-1).abs().fillna(0)
        if tmp_df.time_diff.max() > time_thres:
            discontinous[i] = tmp_df.time_diff.max() 
    return discontinous

#print(get_discontinuous_trip(df, time_thres=10).keys())
trip_dict = get_discontinuous_trip(df, time_thres=10)
trips_to_process = trip_dict.keys()
# trip_dict
# trips_to_process


#For the discontinuous trips, we only preserve the longer (more complete) parts, delete the shorter parts.

def find_drop_times(df, long_trips):
    to_drop = []
    for trip_id in long_trips:
        times = list(df[df.trip_id==trip_id].Time)
        count = 0
        i = 0
        # the jump in time is less than 10 mins
        while((i<len(times)-2) and ((times[i+1]-times[i])< 10)):
            count = count+1
            i = i+1
        # if the more complete trip happens earlier
        if count < len(times)/2:
            drop_time = times[0:count+1]
        elif count+1 == len(times)-1:
            drop_time = [times[-1]]
        else:
            drop_time = times[count+1:-1]
        to_drop = to_drop + drop_time
    return to_drop

# drop the shorter parts in the discontinuous trips
time_to_drop = find_drop_times(df, trip_dict)
df = df[~df.Time.isin(time_to_drop)]

# discountousTrips = get_discontinuous_trip(df, time_thres=10).keys()
# discountousTrips
#[307, 325, 379, 1105, 1199, 1686, 1765, 1909, 2523, 2639, 3084, 4023]


# drop shorter parts second time to avoid any trip with mutltiple discontinues
# trip_dict = get_discontinuous_trip(df, time_thres=10)
trips_to_process = get_discontinuous_trip(df, time_thres=10).keys()
# trips_to_process
time_to_drop = find_drop_times(df, trip_dict)
# time_to_drop
df = df[~df.Time.isin(time_to_drop)]
discountinousTrips = get_discontinuous_trip(df, time_thres=10) #3084
# discountinousTrips

# plt.scatter(df[df.trip_id==3084].LONGITUDE, df[df.trip_id==3084].LATITUDE, s=2, c=df[df.trip_id==3084].Time, cmap="BrBG")

df = df[~df['trip_id'].isin(discountinousTrips)] # delete discountinous trips (id: 3084)

# 3084 is a wierd trip, just remove it from the dataset
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.show()
# df = df[df.trip_id!=3075]
#df[df.trip_id == 307][['Dati','Time','LONGITUDE','LATITUDE','SOG','trip_id']]


# #After slicing, we will have trips that starts or ends too far away from the bay.\
# # In other words, they are incomplete trips. Remove these incomplete trips


def incomplete_trips(df, thresh=1e-3):
    incomplete_trips = []
    trips = list(df.trip_id.unique())
    trips.remove(0)
    for trip_id in trips:
        tmp_df = df[df.trip_id==trip_id].reset_index()
        start_H_dist = (tmp_df.iloc[0].LONGITUDE - H_long)**2 + (tmp_df.iloc[0].LATITUDE - H_lat)**2
        start_N_dist = (tmp_df.iloc[0].LONGITUDE - N_long)**2 + (tmp_df.iloc[0].LATITUDE - N_lat)**2
        start_bay_dist = min(start_H_dist, start_N_dist)
        end_H_dist = (tmp_df.iloc[-1].LONGITUDE - H_long)**2 + (tmp_df.iloc[-1].LATITUDE - H_lat)**2
        end_N_dist = (tmp_df.iloc[-1].LONGITUDE - N_long)**2 + (tmp_df.iloc[-1].LATITUDE - N_lat)**2
        end_bay_dist = min(end_H_dist, end_N_dist)
        travel_dist = (tmp_df.iloc[0].LONGITUDE - tmp_df.iloc[-1].LONGITUDE)**2 + \
            (tmp_df.iloc[0].LATITUDE - tmp_df.iloc[-1].LATITUDE)**2
        if (start_bay_dist>thresh) | (end_bay_dist>thresh) | (travel_dist<.15):
            incomplete_trips.append(trip_id)
    df = df[~df.trip_id.isin(incomplete_trips)]
    print(incomplete_trips)
    return df
df = incomplete_trips(df, 1e-3)


# df.groupby(df.trip_id).count().Dati[df.groupby(df.trip_id).count().Dati<60]
# #3111, 3481
# print(df.groupby(df.trip_id).count().Dati[df.groupby(df.trip_id).count().Dati<90])
# # same as <60


# id = 3481
# plt.scatter(df[df.trip_id==id].LONGITUDE, df[df.trip_id==id].LATITUDE, s=2, c=df[df.trip_id==id].Time, cmap="BrBG")
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.show()

shortTrips = df.groupby(df.trip_id).count().Dati[df.groupby(df.trip_id).count().Dati<60].copy()
shortTrips # 897, 936, 996, 1050, 1052, 1247, 1315, 1624
# df[df.trip_id == 1247]

#df[(df.Time >= 466395.0) & (df.Time < 466403.0)][['Dati','Time','SOG','LONGITUDE','LATITUDE','trip_id']]
#df[(df.Time >= 423479)][['Dati','Time','SOG','LONGITUDE','LATITUDE','trip_id']]

    
# remove all trips has too less data
#df = df[~df.trip_id.isin([3111,3481])]
df = df[~df['trip_id'].isin(shortTrips.index)]




# # trip 3111 and 3481 are another two special cases, they do not have any jump in time, \
# # in other words, no missing data in between, but they have a location jump\
# # remove them from the data set

print(df.groupby(df.trip_id).count().Dati[df.groupby(df.trip_id).count().Dati>150]) # return 3742


# plt.scatter(df[df.trip_id==3742].LONGITUDE, df[df.trip_id==3742].LATITUDE, s=2, c=df[df.trip_id==3742].Time, cmap="BrBG")
# 3742 is a special case where a detour happened (Dati.count is 162)

print(df.groupby(df.trip_id).count().Dati.min(), df[df.trip_id!=0].groupby(df.trip_id).count().Dati.max())
# return 95,162



# #Another type of abnormal trips are those that has the same starting and end point. \
# # This could be caused by two or more trips classfied into one (long_trips), or a short movement misclassfied into a single trip (short_trips). \
# # We nolonger have these type of trips after previous steps to clean up the data.

def find_round_trip(df):
    short_trip = []
    long_trip = []
    trips = list(df.trip_id.unique())
    trips.remove(0)
    for i in trips:
        tmp_df = df[df.trip_id == i].reset_index()
        start_long, start_lat = tmp_df.iloc[0].LONGITUDE,tmp_df.iloc[0].LATITUDE
        end_long, end_lat = tmp_df.iloc[-1].LONGITUDE,tmp_df.iloc[-1].LATITUDE
        if (abs(start_long - end_long) < 0.005) and (abs(start_lat- end_lat) < 0.005):
            min_long, min_lat = tmp_df.LONGITUDE.min(), tmp_df.LATITUDE.min()
            max_long, max_lat = tmp_df.LONGITUDE.max(), tmp_df.LATITUDE.max()
            if ((max_long-min_long)>0.1) | ((max_lat-min_lat)>0.1):
                long_trip.append(i)
            else:
                short_trip.append(i)
    return short_trip, long_trip
print(find_round_trip(df))

df.trip_id = df.trip_id.astype(int)

print(df.groupby("trip_id").count().Dati.quantile(.01), df.groupby("trip_id").count().Dati.median(), df.groupby("trip_id").count().Dati.quantile(.99))
print(df.groupby(df.trip_id).count().Dati.min(), df[df.trip_id!=0].groupby(df.trip_id).count().Dati.max())

plt.scatter(df.LONGITUDE, df.LATITUDE, s=1)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()








# pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.width',1000)
# print(df[[ 'trip_id','Dati','Time','CountDown','LONGITUDE','LATITUDE']].head(315))
# print(df.columns)

# df.to_csv('~/BCFerryData/queenCsvOut_cleaned_location.csv', index=False)

df.to_csv('data/Features/queenCsvOut_step1.csv', index=False)
