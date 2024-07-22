
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import r2_score, mean_squared_error
from torch.optim.lr_scheduler import StepLR
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel, TimeSeriesTransformerForPrediction
from transformers import AutoformerConfig, AutoformerModel, AutoformerForPrediction



path = "data/Features/feature3.csv"
df = pd.read_csv(path)
df.columns #length 53


#### Index(['Dati', 'Time', 'DEPTH',  'HEADING', 'LATITUDE', 'LONGITUDE',
    #    'SOG', 'SOG_SPEEDLOG_LONG',
    #    'SOG_SPEEDLOG_TRANS',  'STW',  'WIND_ANGLE', 'WIND_SPEED',
    #    'WIND_ANGLE_TRUE', 'WIND_SPEED_TRUE', 'trip_id', 'MODE',
    #    'direction', 'wind_force', 'wind_direc', 'effective_wind_factor',
    #    'effective_wind', 'datetime',
    #    'weekday', 'Schedule', 'ScheduleType', 'countDown', 'season', 'current',
    #    'pressure', 'rain', 'snowfall', 'weathercode', 'holiday', 'is_weekday',
    #    'adversarial'],
    #   dtype='object')###



df["direction"] = df["direction"].apply(lambda x: 1 if x=="N-H" else 0)
df["MODE"] = df["MODE"].apply(lambda x: x-1)
# df = df[df["adversarial"]==0].drop("adversarial", axis=1)
# df = df[df["direction"]==1].drop("direction", axis=1)
df["Time2"] = df.groupby("trip_id")["Time"].rank(method="first", ascending=True)
df["is_weekday"] = df["is_weekday"].apply(lambda x: 1 if x==True else 0)

df["change_x_factor"] = np.cos((df.HEADING+90) * np.pi / 180)
df["change_y_factor"] = np.sin((df.HEADING-90) * np.pi / 180)
df["goal_long"] = df.direction.apply(lambda x: 0.9965111208024382 if x==1 else 0.0023259194650222526)
df["goal_lat"] = df.direction.apply(lambda x: 0.7729570345408661 if x==1 else 0)
df["prev_HEADING"] = df.HEADING.shift(periods=1)
df["turn"] = df.HEADING - df.prev_HEADING
# df["place_long"] = df["change_x_factor"] * df["SOG"]

df["POWER"] = (df["POWER_1"]+df["POWER_2"])/2
df["SPEED"] = (df["SPEED_1"]+df["SPEED_2"])/2
df["THRUST"] = (df["THRUST_1"]+df["THRUST_2"])/2
df["TORQUE"] = (df["TORQUE_1"]+df["TORQUE_2"])/2
df["PITCH"] = (df["PITCH_1"]+df["PITCH_2"])/2
df["resist_ratio"] = (df["resist_ratio1"]+df["resist_ratio2"])/2
df["FLOWTEMPA"] = (df["ENGINE_1_FLOWTEMPA"]+df["ENGINE_2_FLOWTEMPA"])/2
df["FC"] = (df["ENGINE_1_FUEL_CONSUMPTION"]+df["ENGINE_2_FUEL_CONSUMPTION"])/2

df = df.drop(['PITCH_1', 'PITCH_2', 'POWER_1', 'POWER_2', 'SOG_SPEEDLOG_LONG',
       'SOG_SPEEDLOG_TRANS',  'THRUST_1',
       'THRUST_2', 'TORQUE_1', 'TORQUE_2', 'datetime',
       'resist_ratio1', 'resist_ratio2', 'SPEED_1', 'SPEED_2',
       'ENGINE_1_FLOWTEMPA', 'ENGINE_2_FLOWTEMPA',
       "ENGINE_1_FUEL_CONSUMPTION", "ENGINE_2_FUEL_CONSUMPTION"], axis=1)

df["dt"] = pd.to_datetime(df.Schedule, format='%Y-%m-%d %H:%M:%S')
# df["hour"] = df.dt.apply(lambda x: x.hour/24)
df["departure_hour"] = df.dt.apply(lambda x: x.hour/24)

# get one_hot for column
def one_hot(df, cols, normalize = True):
    for col in cols:
        dummy = pd.get_dummies(df[col],prefix=col, drop_first=True)
        columns = dummy.columns
        dummy[col] = 0
        i = 1
        for x in columns:
            dummy[col] = dummy[col] + dummy[x]*i
            i = i*2
        max = dummy[col].max()
        if normalize:
            df[col] = dummy[col] / max
        else:
            df[col] = dummy[col]
    return df



minmax_scaler = MinMaxScaler((0,1))

transform_cols = [ 'current', 'rain', 'snowfall', "pressure", 'wind_force', "resist_ratio",
       'FC', "LATITUDE", 'LONGITUDE', 'SOG', "DEPTH", "SPEED"]
# df[transform_cols]
minmax_scaler = minmax_scaler.fit(df[transform_cols].values)
import pickle
pickle.dump(minmax_scaler, open('minmax_scaler_2.pkl', 'wb'))
df[transform_cols] = minmax_scaler.transform(df[transform_cols])

df["prev_SOG"] = df.SOG.shift(periods=1)
df["acceleration"] = ((df.SOG - df.prev_SOG))
df["distance"] = ((df["goal_long"]-df["LONGITUDE"])**2 + \
                             (df["goal_lat"]-df["LATITUDE"])**2 )**0.5


df = one_hot(df, ["season", "weathercode", "wind_direc"], normalize = True)
# df = df[df["adversarial"]==0]


df.to_csv("data/Features/feature4.csv", index=False)
