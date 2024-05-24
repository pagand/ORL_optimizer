import pandas as pd
import numpy as np
import datetime
from pykalman import KalmanFilter

df = pd.read_csv("~/BCFerryData/queenCsvOut_cleaned_location.csv")

df.drop(["CARGO", "CARGO_PAX", "PAX", 'TRACK_MADE_GOOD', 
        'ENGINE_1_FLOWRATE', 'ENGINE_1_FLOWRATEA', 'ENGINE_1_FLOWRATEB', 'ENGINE_1_FLOWTEMPB',
        'ENGINE_2_FLOWRATE', 'ENGINE_2_FLOWRATEA', 'ENGINE_2_FLOWRATEB','ENGINE_2_FLOWTEMPB'
        ], inplace=True, axis=1)

df["WIND_ANGLE"] = df["WIND_ANGLE"].apply(lambda x: x-360 if x>360 else x) 
df = df[df.trip_id!=0]



cols = list(df.columns)
remove_list = ["Dati", "Time", "HEADING", "LONGITUDE", "LATITUDE", "WIND_ANGLE", "WIND_ANGLE_TRUE", "WIND_SPEED",
               "trip_id", "DEPTH", "PITCH_1", "PITCH_2"]
for col in remove_list:
    cols.remove(col)


for col in cols:
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    IQR = q3 - q1
    lower = q1 - abs(1.5 * IQR)
    upper = q3 + abs(1.5 * IQR)
    outlier_count = df[(df[col]<lower) | (df[col]>upper)].Dati.count()
    if outlier_count > 0:
        print(col)
        print(outlier_count)
        print("\n")


# Flow Temp
# Low flow temp happens only when SOG is low, and at a time where outside temperature could be low Seems reasonable

col = "ENGINE_1_FLOWTEMPA"
q1 = df[col].quantile(.25)
q3 = df[col].quantile(.75)
IQR = q3 - q1
lower = q1 - abs(1.5 * IQR)
upper = q3 + abs(1.5 * IQR)
tmp = df[(df[col]<lower) | (df[col]>upper)][["ENGINE_1_FLOWTEMPA", "ENGINE_2_FLOWTEMPA", "Dati", "SOG"]]
#tmp[tmp.SOG > 1]
print(tmp)

# Fuel Consumption

col = "ENGINE_1_FUEL_CONSUMPTION"
q1 = df[col].quantile(.25)
q3 = df[col].quantile(.75)
IQR = q3 - q1
lower = q1 - abs(1.5 * IQR)
upper = q3 + abs(1.5 * IQR)
tmp = df[(df[col]<lower) | (df[col]>upper)]
print(IQR, q1, q3, upper, lower)
tmp[["ENGINE_1_FUEL_CONSUMPTION", "ENGINE_2_FUEL_CONSUMPTION", "POWER_1", "POWER_2", "SOG", "trip_id"]]

# Power_2
# observing from visluazation, the range for power 1 and 2 is similar, but power 2 has a lot of outliers.
# This is because power 2 has more 0's, (i.e. engine 2 is used less than engine 1).
# No need to remove these outliers

# Rate Of Turns
# outliers in rate of turn is also caused by 0 values.
# Inaccurate 0's, drop rate of turn.
df[df.RATE_OF_TURN==0][["Time", "HEADING", "RATE_OF_TURN", "trip_id"]].groupby(["trip_id"]).count()
df[df.RATE_OF_TURN==0].groupby(["trip_id"]).count()[df[df.RATE_OF_TURN==0].groupby(["trip_id"]).count()>30].dropna()["Dati"]
#df[df.trip_id==4050][["Time", "HEADING", "RATE_OF_TURN", "trip_id"]]
df.drop(["RATE_OF_TURN"], axis=1, inplace=True)

# SOG

col = "SOG"
q1 = df[col].quantile(.25)
q3 = df[col].quantile(.75)
IQR = q3 - q1
lower = q1 - abs(1.5 * IQR)
upper = q3 + abs(1.5 * IQR)
print(lower, upper)
print(df.SOG.min(), df.SOG.max()) #min = 0.0833, max = 21.875



# SOG_SPEEDLOG_TRANS
# the extreme values in SOG_SPEEDLOG_TRANS seems reasonable when comparing to corresponding SOG and SOG_SPEEDLOG_LONG

col = "SOG_SPEEDLOG_TRANS"
q1 = df[col].quantile(.25)
q3 = df[col].quantile(.75)
IQR = q3 - q1
lower = q1 - abs(1.5 * IQR)
upper = q3 + abs(1.5 * IQR)
print(lower, upper)
print(df.SOG_SPEEDLOG_TRANS.min(), df.SOG_SPEEDLOG_TRANS.max())

tmp = df[(df[col]<lower) | (df[col]>upper)].copy()
#tmp
tmp["computed_SOG"] = df.SOG_SPEEDLOG_TRANS**2 + df.SOG_SPEEDLOG_LONG**2
tmp["SOG_squred"] = df.SOG ** 2
tmp[(tmp.computed_SOG - tmp.SOG_squred)>3] # empty data set

# SPEED_1
# the extreme values in speed_1 seems unreasonable when comparing to related fields remove them and impute later.

col = "SPEED_1"
q1 = df[col].quantile(.25)
q3 = df[col].quantile(.75)
IQR = q3 - q1
lower = q1 - abs(1.5 * IQR)
upper = q3 + abs(1.5 * IQR)
tmp = df[(df[col]<lower) | (df[col]>upper)][["SPEED_1", "SPEED_2", "POWER_1", "POWER_2"]]
# tmp # return 168304
df.loc[tmp.index, "SPEED_1"] = np.nan 


# STW
# Similar to SOG, outliers are lower STWs, these low speed values are reasonable to have.
col = "STW"
q1 = df[col].quantile(.25)
q3 = df[col].quantile(.75)
IQR = q3 - q1
lower = q1 - abs(1.5 * IQR)
upper = q3 + abs(1.5 * IQR)
print(lower, upper)
print(df.STW.min(), df.STW.max())


# WIND_SPEED_TRUE

col = "WIND_SPEED_TRUE"
q1 = df[col].quantile(.25)
q3 = df[col].quantile(.75)
IQR = q3 - q1
lower = q1 - abs(1.5 * IQR)
upper = q3 + abs(1.5 * IQR)
tmp = df[(df[col]<lower) | (df[col]>upper)][["WIND_SPEED", "WIND_SPEED_TRUE", "WIND_ANGLE", "WIND_ANGLE_TRUE"]]
tmp

outlier_indexes = list(tmp.index)
for index in outlier_indexes:
    df.loc[index, "WIND_SPEED"] = np.nan
    df.loc[index, "WIND_SPEED_TRUE"] = np.nan





#df = df.to_csv("../../df_outlier_removed.csv", index=False)
df = df.to_csv("~/BCFerryData/df_outlier_removed.csv", index=False)
df = pd.read_csv("~/BCFerryData/df_outlier_removed.csv")


#### IMPUTING MISSING DATA #####

# kalman filter to impute missing values
def impute_missing_values(data, transition_matrices, observation_matrices, transition_covariance,
                          observation_covariance, initial_state_mean, initial_state_covariance):
    kf = KalmanFilter(transition_matrices=transition_matrices,
                      observation_matrices=observation_matrices,
                      transition_covariance=transition_covariance,
                      observation_covariance=observation_covariance,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance)
    
    # Create a mask indicating missing values
    mask = np.isnan(data)
    filtered_state_means, _ = kf.filter(data)
    
    # Replace missing values with imputed values
    imputed_data = data.copy()
    imputed_data[mask] = filtered_state_means[mask]
    
    return imputed_data

# Add an empty line for each missing time within trips

def missing_time(df):
    missing_dict = {}
    for trip_id in (df.trip_id.unique()):
        tmp_df = df[df.trip_id==trip_id]
        fill_in = tmp_df.Time.max()
        # tmp_df["time_up"] = tmp_df["Time"].shift(periods=-1, fill_value=fill_in)
        # tmp_df["time_diff"] = tmp_df["time_up"] - tmp_df["Time"]
        tmp_df["time_diff"] = tmp_df["Time"].diff(-1).abs().fillna(0)
        missing = tmp_df[tmp_df["time_diff"]>1]
        if len(missing)>0:
            missing_list = []
            for i in list(missing.index):
                start = int(missing.loc[i, "Time"]+1)
                end = int(missing.loc[i, "Time"]+missing.loc[i, "time_diff"])
                missing_list = missing_list + [ x for x in range(start, end)]
            missing_dict[trip_id] = missing_list
            # missing_time = missing_time + list(tmp_df[tmp_df["time_diff"]>1].Time)
    return missing_dict

missing = missing_time(df).copy()
len(missing)

for trip_id in missing.keys():
    for time in missing[trip_id]:
        df.loc[len(df)] = {"Time":time, "trip_id":trip_id}

df = df.sort_values("Time").reset_index(drop=True)

#df.columns

# Imputing Data

def impute_Dati(df):#
    missing = (df[pd.isna(df.Dati)].index)
    for i in missing:
        previous_dt = pd.to_datetime(df.loc[i-1].Dati, format='%y%m%d_%H%M%S')
        dt = (previous_dt +  datetime.timedelta(minutes=1)).strftime('%y%m%d_%H%M%S')
        df.loc[i, "Dati"] = dt

impute_Dati(df)

def impute_mode(df, cols):
    for col in cols:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode)
        
impute_mode(df, ["DEPTH"])

def impute_mean_within_trips(df, cols):
    for col in cols:
        na_trips = df[df[col].isna()].trip_id.unique()
        for trip in na_trips:
            tmp_df = df[df.trip_id==trip]
            missing = list(tmp_df[tmp_df[col].isna()].index)
            mean = tmp_df[col].mean()
            df.loc[missing, col] = mean

impute_mean_within_trips(df, ["HEADING", "WIND_SPEED", "WIND_SPEED_TRUE", "WIND_ANGLE", "WIND_ANGLE_TRUE"])

df.isna().sum()[df.isna().sum()>0]

# df[(df.LATITUDE.isna()) & (df.THRUST_1.notna() | (df.THRUST_2.notna())) & df.SOG != 0][['trip_id','SOG','THRUST_1','THRUST_2']]
# df[(df.trip_id == 458)]
# df[(df.THRUST_1 == 0)]

def impute_missing(df, cols):
    for col in cols:
        na_trips = df[df[col].isna()].trip_id.unique()
        for trip in na_trips:
            tmp_df = df[df.trip_id==trip]
            missing = list(tmp_df[tmp_df[col].isna()].index)
            prev_val = df.loc[missing[0]-1, col]
            after_val = df.loc[missing[-1]+1, col]
            if len(missing) > 1:
                impute_diff = (after_val- prev_val)/(missing[-1]-missing[0]+1)
            else:
                impute_diff = (after_val- prev_val)/2
            for i in range(len(missing)):
                df.loc[missing[i], col] = prev_val + impute_diff*(i+1)

impute_missing(df, df.columns)

df.isna().sum()[df.isna().sum()>0]

# add countdown
TRIP_DURATION = 100 
df["start_time"] = df.groupby("trip_id")['Time'].transform('min')
df["CountDown"] = TRIP_DURATION - (df['Time'] - df['start_time'])
df.loc[df['trip_id'] == 0, 'CountDown'] = 0
df.drop(['start_time'], axis=1, inplace=True)

df.isna().sum()[df.isna().sum()>0]


df.to_csv('~/BCFerryData/df_naive_impute.csv', index=False)


# pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.width',1000)
# print(tmp.head(20))



