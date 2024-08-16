
import os
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
from transformers import InformerForPrediction, InformerConfig
import torch.optim as optim
import wandb
import model
from config import VesselConfig as config


# path = "data/Features/feature4.csv"
path = "data/Features/feature2.csv"
df = pd.read_csv(path)

df= df[df["adversarial"] == 0]

df.iloc[0, df.columns.get_loc('prev_HEADING')] = df.iloc[1, df.columns.get_loc('prev_HEADING')]
df.iloc[0, df.columns.get_loc('prev_SOG')] = df.iloc[1, df.columns.get_loc('prev_SOG')]
df.iloc[0, df.columns.get_loc('turn')] = df.iloc[1, df.columns.get_loc('turn')]
df.iloc[0, df.columns.get_loc('acceleration')] = 0


columns = df.columns.drop("countDown")
shifted_df = df.copy()
shifted_df[columns] = df[columns].shift(periods=5)


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# batch_size=256
# sequence_length = 25
# context_length = 24
# prediction_horizon = 5 #10
# data_len = 90


# # initialize the model
# time_feature = ["countDown"]
# dynamic_real_feature = [ "SPEED", "HEADING", "MODE", "turn", "acceleration",
#        'current', 'rain', 'snowfall', 'wind_force', 'wind_direc',
#         "resist_ratio","change_x_factor", "change_y_factor"]# 
# static_categorical_feature = ["is_weekday", 'direction',"season", "departure_hour"] # ScheduleType #"adversarial"
# y_cols = ["FC","SOG","LONGITUDE","LATITUDE"]

batch_size=config.batch_size
sequence_length = config.sequence_length
context_length = config.context_length
prediction_horizon = config.prediction_horizon
data_len = config.data_len


# initialize the model
time_feature = config.time_feature
dynamic_real_feature = config.dynamic_real_feature
static_categorical_feature = config.static_categorical_feature
y_cols = config.y_cols

# all_copies = pd.DataFrame()

# for i in range(1000):
#     # Add random noise to numerical data
#     noisy_data = df[df["trip_id"] == 4].copy()
#     cols = dynamic_real_feature + y_cols
#     cols.remove("countDown")
#     cols.remove("MODE")
#     for col in cols:
#         # Adding noise: mean 0, std 0.05 * standard deviation of the column
#         noise = np.random.normal(0, 0.05 * noisy_data[col].std(), size=noisy_data[col].shape)
#         noisy_data[col] += noise
    
#     # Append the noisy data to the all_copies DataFrame
#     noisy_data["trip_id"] = 4 + i
#     all_copies = all_copies.append(noisy_data, ignore_index=True)

# df = all_copies

# data class
class vessel_data(Dataset):
    def __init__(self, train = True, test=False, train_test_split = 0.7, rand_seed=1, config=None):
        ##########################inputs##################################
        #data_dir(string) - directory of the data#########################
        #size(int) - size of the images you want to use###################
        #train(boolean) - train data or test data#########################
        #train_test_split(float) - the portion of the data for training###
        #augment_data(boolean) - use data augmentation or not#############
        super(vessel_data, self).__init__()
        # todo
        #initialize the data class
        trips = list(df.trip_id.unique())
        self.train = train
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.testTripId = []

        # train_test_split
        random.seed(rand_seed)
        train_size = int(np.ceil(len(trips)*train_test_split))
        train_trips = random.sample(trips, k=train_size)
        if train:
            self.trips_id = train_trips
            # self.trips_id = train_trips[0:2]
        else:
            test_trips = [ x for x in trips if x not in train_trips]
            valid_trips = random.sample(test_trips, k = int(np.ceil(len(trips)*((1-train_test_split)*.7))))
            if test==False:
                self.trips_id = valid_trips
            else:
                self.trips_id = [ x for x in test_trips if x not in valid_trips]
                self.testTripId = self.trips_id
        # self.starting_dict = self.get_starting(df)
    
    def get_testTripId(self):
        return self.testTripId

    # convert a df to tensor
    def df_to_tensor(self, df):
        numpy = df.to_numpy(dtype="double")
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        # return torch.from_numpy(df.values.astype("float")).float().to(device)
        return torch.from_numpy(numpy).float().to(device)
    
    def __getitem__(self, idx):
        #load corresponding trip id from index idx of your data
        # trip_id = (self.trips_id[idx//2])
        trip_id = self.trips_id[idx]
        # start = self.starting_dict[trip_id]
        # start_idx = self.starting_dict[trip_id] + (idx%3)*(self.prediction_horizon+self.sequence_length)
        
        # start_auto = df[(df["trip_id"] == trip_id) & (df["MODE"] == 0)].index[0]
        end = df[(df["trip_id"] == trip_id) & (df["MODE"] == 0)].index[-1] +1
        # end = 64 + start
        # df[(df["trip_id"] == 2236) & (df["MODE"] == 0)].index[0]
        # data = df[df.trip_id==trip_id][start:end].reset_index(drop=True)
        # shifted_data = shifted_df[df.trip_id==trip_id][start:end].reset_index(drop=True)
        
        start = df[df["trip_id"] == trip_id].index[0]
        # start = max(start, start_auto - config.sequence_length)
        data = df[start:end].reset_index(drop=True)
        shifted_data = shifted_df[start:end].reset_index(drop=True)

        # print(start, end, len(data))
        if(len(data) < data_len):
            diff = data_len - len(data)
            # data["done"] = 0
            # data_done = data[end-diff:end]
            # data_done["done"] = 1
            data = pd.concat([data, data[-diff:]]).reset_index(drop=True)
            shifted_data = pd.concat([shifted_data, shifted_data[-diff:]]).reset_index(drop=True)
            # print(len(data))
            # data = pd.concat([data, data_done]).reset_index(drop=True)

        # print(len(data))

        # starting = random.randint(0, math.floor((len(data)-data_len)/2))
        starting = 0
        # if (idx%2):
        #     starting = len(data)-data_len

        data = data.iloc[starting:starting+data_len]
        # print(len(data))

        values = data[y_cols]
        time_features = data[time_feature + dynamic_real_feature]
        static_categorical_features = data.iloc[0][static_categorical_feature]
        # future_time_features = shifted_data.iloc[starting:starting+data_len][time_feature + dynamic_real_feature]
        future_time_features = shifted_data.iloc[starting:starting+data_len][time_feature + dynamic_real_feature]
        actions = data[["SPEED", "HEADING", "MODE", "turn"]]
        end_auto = end - start
        return self.df_to_tensor(values), self.df_to_tensor(time_features), self.df_to_tensor(static_categorical_features), self.df_to_tensor(future_time_features), self.df_to_tensor(actions), end_auto

    def __len__(self):
        # return len(self.trips_id)*2
        return len(self.trips_id)



testset = vessel_data(train = False, test=True, train_test_split = 0.8, rand_seed=2)
def get_testTripId():
    return testset.get_testTripId()

testTripId = get_testTripId()






# # data class
# class vessel_data(Dataset):
#     def __init__(self, train = True, test=False, train_test_split = 0.7, rand_seed=1, config=None):
#         ##########################inputs##################################
#         #data_dir(string) - directory of the data#########################
#         #size(int) - size of the images you want to use###################
#         #train(boolean) - train data or test data#########################
#         #train_test_split(float) - the portion of the data for training###
#         #augment_data(boolean) - use data augmentation or not#############
#         super(vessel_data, self).__init__()
#         # todo
#         #initialize the data class
#         trips = list(df.trip_id.unique())
#         self.train = train
#         self.sequence_length = sequence_length
#         self.prediction_horizon = prediction_horizon
#         self.testTripId = []

#         # train_test_split
#         random.seed(rand_seed)
#         train_size = int(np.ceil(len(trips)*train_test_split))
#         train_trips = random.sample(trips, k=train_size)
#         if train:
#             self.trips_id = train_trips
#             # self.trips_id = train_trips[0:2]
#         else:
#             test_trips = [ x for x in trips if x not in train_trips]
#             valid_trips = random.sample(test_trips, k = int(np.ceil(len(trips)*((1-train_test_split)*.7))))
#             if test==False:
#                 self.trips_id = valid_trips
#             else:
#                 self.trips_id = [ x for x in test_trips if x not in valid_trips]
#                 self.testTripId = self.trips_id
#         # self.starting_dict = self.get_starting(df)
    
#     def get_testTripId(self):
#         return self.testTripId

#     # convert a df to tensor
#     def df_to_tensor(self, df):
#         numpy = df.to_numpy(dtype="double")
#         if torch.cuda.is_available():
#             device = torch.device('cuda:0')
#         else:
#             device = torch.device('cpu')
#         # return torch.from_numpy(df.values.astype("float")).float().to(device)
#         return torch.from_numpy(numpy).float().to(device)
    
#     def __getitem__(self, idx):
#         #load corresponding trip id from index idx of your data
#         trip_id = (self.trips_id[idx//2])
#         # trip_id = self.trips_id[idx]
#         # start = self.starting_dict[trip_id]
#         # start_idx = self.starting_dict[trip_id] + (idx%3)*(self.prediction_horizon+self.sequence_length)
#         start = df[(df["trip_id"] == trip_id) & (df["MODE"] == 0)].index[0]
#         end = df[(df["trip_id"] == trip_id) & (df["MODE"] == 0)].index[-1] +1
#         # end = 64 + start
#         # df[(df["trip_id"] == 2236) & (df["MODE"] == 0)].index[0]
#         # data = df[df.trip_id==trip_id][start:end].reset_index(drop=True)
#         # shifted_data = shifted_df[df.trip_id==trip_id][start:end].reset_index(drop=True)
#         data = df[start:end].reset_index(drop=True)
#         shifted_data = shifted_df[start:end].reset_index(drop=True)

#         starting = random.randint(0, math.floor((len(data)-data_len)/2))
#         # starting = 0
#         # if (idx%2):
#         #     starting = len(data)-91

#         # starting = 0
       
#         data = data.iloc[starting:starting+data_len]
        
#         values = data[y_cols]
#         time_features = data[time_feature + dynamic_real_feature]
#         static_categorical_features = data.iloc[0][static_categorical_feature]
#         # future_time_features = shifted_data.iloc[starting:starting+data_len][time_feature + dynamic_real_feature]
#         future_time_features = shifted_data.iloc[starting:starting+data_len][time_feature + dynamic_real_feature]
#         actions = data[["SPEED", "HEADING", "MODE", "turn"]]
#         return self.df_to_tensor(values), self.df_to_tensor(time_features), self.df_to_tensor(static_categorical_features), self.df_to_tensor(future_time_features), self.df_to_tensor(actions)

#     def __len__(self):
#         return len(self.trips_id)*2
#         # return len(self.trips_id)
