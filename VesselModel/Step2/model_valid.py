
import math
import pandas as pd
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
# from train_v1 import get_testTripId,evaluate
from data import get_testTripId
from config import VesselConfig as config
import wandb
import model


class testTrip():
    def __init__(self, data, models_file_path):
        # self.tripId = tripId
        # self.df = df
        # self.data = data
        # start = data[data["MODE"] == 0].index[0]
        start = data.index[0]
        end = data[data["MODE"] == 0].index[-1] + 1

        self.data = data[start:end].reset_index(drop=True)
        self.max_steps = len(self.data)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        # load forecasting models
        self._load_model(models_file_path)
        self._set_eval()

    def _load_model(self,filepath):
        self.time_feature = config.time_feature
        self.dynamic_real_feature = config.dynamic_real_feature
        self.static_categorical_feature = config.static_categorical_feature
        self.y_cols = config.y_cols
       
        tfconfig = InformerConfig.from_pretrained("huggingface/informer-tourism-monthly", prediction_length=config.prediction_horizon,
                context_length=config.context_length, input_size=len(self.y_cols), num_time_features=len(self.time_feature),
                num_dynamic_real_features = len(self.dynamic_real_feature), num_static_real_features = len(self.static_categorical_feature),
                lags_sequence=[1], num_static_categorical_features=0, feature_size=31)

        tf_model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly",
                                                config=tfconfig, ignore_mismatched_sizes=True).to(self.device)

        self.gru = model.GRU_update(4, hidden_size=375, output_size=4, num_layers=1, prediction_horizon=5, device=self.device).to(self.device)


        self.tf= tf_model.float()
        self.tf.load_state_dict(torch.load(filepath[0], map_location=torch.device(self.device)))
        self.gru.load_state_dict(torch.load(filepath[1], map_location=torch.device(self.device)))
        
        # approach 2
        # self.model = model.vesselModel(self.tf, self.gru)
        # self.model.load_state_dict(torch.load(filepath[0], map_location=torch.device(self.device)))

    def _set_eval(self):
        self.tf.eval()
        self.gru.eval()
        # self.model.eval()

        self.current_step = starting_step

    def _step(self):
        self.past_feature = self.data[self.current_step-25:self.current_step][self.time_feature + self.dynamic_real_feature].copy()
        if (self.current_step == starting_step):
            self.past_values = self.data[self.current_step-25:self.current_step][self.y_cols].copy()
            self.static_values = self.data.iloc[0][self.static_categorical_feature].copy()
            
        self.future_feature = self.past_feature[-5:].copy()
        
        self.future_feature[self.future_feature.columns[-1]] = self.future_feature[self.future_feature.columns[-1]] - 1
        # predict
        # fc, sog, longitude, latitude = self._predict(self.past_values, self.past_feature, self.future_feature, self.static_values, self.model)
        fc, sog, longitude, latitude = self._predict(self.past_values, self.past_feature, self.future_feature, self.static_values, self.tf, self.gru)
        # update past values for next step
        self.past_values = self.past_values[1:self.current_step]
        self.past_values.append(pd.Series(), ignore_index=True)
        self.past_values.loc[self.current_step] = fc, sog, longitude, latitude

        # dict = {"countDown": self.data.iloc[self.current_step]["countDown"], 
        #         "FC": fc, "SOG": sog, "LONGITUDE": longitude, "LATITUDE": latitude}
        dict = { "time ": self.data.iloc[self.current_step]["Time2"],
                "countDown": self.data.iloc[self.current_step]["countDown"], 
                "FC": fc, "SOG": sog, "LONGITUDE": longitude, "LATITUDE": latitude,
                "act_FC": self.data.iloc[self.current_step]["FC"], "act_SOG": self.data.iloc[self.current_step]["SOG"],
                "act_LONGITUDE": self.data.iloc[self.current_step]["LONGITUDE"], "act_LATITUDE": self.data.iloc[self.current_step]["LATITUDE"]
        }
        wandb.log(dict)
        
        return fc, sog, longitude, latitude

    def _predict(self, past_values, past_feature, future_feature, static_feature, 
                 #model,
                 tf_model, gru_model
                 ):
        future_feature = torch.from_numpy(np.expand_dims(future_feature, 0)).float().to(self.device)
        past_values = torch.from_numpy(np.expand_dims(past_values, 0)).float().to(self.device)
        past_feature = torch.from_numpy(np.expand_dims(past_feature, 0)).float().to(self.device)
        static_feature = torch.from_numpy(np.expand_dims(np.array(static_feature).astype(float), 0)).float().to(self.device)
        past_observed_mask = torch.ones(past_values.shape).to(self.device)
        
        with torch.no_grad():
            tf_out = tf_model.generate(past_values=past_values, past_time_features=past_feature,
                    static_real_features=static_feature, past_observed_mask=past_observed_mask,
                    future_time_features=future_feature).sequences.mean(dim=1)
            outputs = gru_model(tf_out, past_feature).detach().cpu().numpy()
        return outputs[0,0,0], outputs[0,0,1], outputs[0,0,2], outputs[0,0,3]
    
    def run(self):
        self.pred_fc = []
        self.pred_sog = []
        self.pred_longitude = []
        self.pred_latitude = []
        while self.current_step < self.max_steps:
            fc, sog, longitude, latitude = self._step()
            self.current_step += 1
            self.pred_fc.append(fc)
            self.pred_sog.append(sog)
            self.pred_longitude.append(longitude)
            self.pred_latitude.append(latitude)
            # if self.current_step == self.max_steps:
            #     print("Trip done")
            #     break
        print("Trip finished")
        return self.pred_fc, self.pred_sog, self.pred_longitude, self.pred_latitude
    
    def max_step(self):
        return self.max_steps
    

# path = "data/Features/feature2.csv"
path = "data/Features/feature3.csv"
df = pd.read_csv(path)


starting_step = 25
load_iter = config.iter
load_cp = config.best_epoch
# model_name = "vesselModel_Iter_{}".format(load_iter)
model_name = "Model_v1_Iter_{}".format(load_iter)

filepath = ("data/Checkpoints/{}/{}_epoch_{}_tf.pt".format(model_name, model_name, load_cp),
            "data/Checkpoints/{}/{}_epoch_{}_gru.pt".format(model_name, model_name, load_cp)
            )

random.seed(None)


wandb.login()
wandb.init(project="Trip Visualization", name="{}_iter{}_epoch{}".format(model_name, load_iter, load_cp))

# testing = True


testTripsIDs = get_testTripId()
print(testTripsIDs)


while True:
    tripId = input("Enter tripId: ")
    try:
        tripId = int(tripId)
    except:
        print("Invalid tripId")
    if tripId not in testTripsIDs:
        print("Invalid tripId")
    data = df[df["trip_id"]==tripId].reset_index(drop=True)
    test_trip = testTrip(data, filepath)
    fc, sog, longitude, latitude = test_trip.run()
    max_step = test_trip.max_step()
    act_fc = data.iloc[starting_step:max_step]["FC"]
    act_sog = data.iloc[starting_step:max_step]["SOG"]
    act_longitude = data.iloc[starting_step:max_step]["LONGITUDE"]
    act_latitude = data.iloc[starting_step:max_step]["LATITUDE"]
    if (len(act_fc) != len(fc)):
        print("Length of actual and predicted fc are not equal")
        break
    mse_fc = mean_squared_error(act_fc, fc)
    mse_sog = mean_squared_error(act_sog, sog)
    mse_longitude = mean_squared_error(act_longitude, longitude)
    mse_latitude = mean_squared_error(act_latitude, latitude)
    r2_fc = r2_score(act_fc, fc)
    r2_sog = r2_score(act_sog, sog)
    r2_longitude = r2_score(act_longitude, longitude)
    r2_latitude = r2_score(act_latitude, latitude)
    print("MSE: " , mse_fc, mse_sog, mse_longitude, mse_latitude)
    # print("R2: ", r2_fc, r2_sog, r2_longitude, r2_latitude)
    break

