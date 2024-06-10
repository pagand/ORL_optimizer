import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt

# file_path = "~/BCFerryData/df_naive_impute.csv"
file_path = "data/df_naive_impute.csv"
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
# df.head(120)
df["MODE_up"] = df["MODE"].shift(periods=-1)
df["MODE_down"] = df["MODE"].shift(periods=1)
df["MODE"] = df[["MODE", "MODE_up", "MODE_down"]].apply(clean_mode,axis=1)
df.drop(["MODE_up", "MODE_down"],axis=1, inplace=True)

df.groupby("MODE").count()

samples = df.trip_id.sample(100)
plt.scatter(df[df.trip_id.isin(samples)].LONGITUDE, df[df.trip_id.isin(samples)].LATITUDE, c = df[df.trip_id.isin(samples)].MODE, s=1)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()


# # Horseshoe Bay
# # 49.3771, -123.2715
# H_lat = 49.3771
# H_long = -123.2715

# # Nanaimo 
# # 49.1936, -123.9554
# N_lat = 49.1936
# N_long = -123.9554


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

direcs_dict = get_direction() 
# direcs_dict
df["direction"] = df["trip_id"].apply(lambda x: direcs_dict[x])


# Add Wind related feature
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


df.to_csv('data/feature_tmp1.csv', index=False)
#df = pd.read_csv('data/feature_tmp1.csv')

