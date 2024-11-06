import os
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
from transformers import InformerForPrediction, InformerConfig, InformerModel
import torch.optim as optim
import wandb
from config import VesselConfig as config



if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


class GRU_update(nn.Module):
    def __init__(self,input_size, hidden_size=1, output_size=4, num_layers=1, prediction_horizon=5, device = device):
        super().__init__()
        self.h = prediction_horizon
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.mlp = nn.Sequential( nn.ReLU(),
                                  nn.Linear(hidden_size, 2048),
                                  nn.Dropout(0.1),
                                  nn.ReLU(),
                                  nn.Linear(2048, output_size))
        self.hx_fc = nn.Linear(2*hidden_size, hidden_size)
        self.device = device

    def forward(self, predicted_values, past_time_features):
        xy = torch.zeros(size=(past_time_features.shape[0], 1, self.output_size)).float().to(self.device)
        hx = past_time_features.reshape(-1, 1, self.hidden_size)
        hx = hx.permute(1, 0, 2)
        out_wp = list()
        for i in range(self.h):
            ins = torch.cat([xy, predicted_values[:, i:i+1, :]], dim=1) # x
            hx, _ = self.gru(ins, hx.contiguous())
            hx = hx.reshape(-1, 2*self.hidden_size)
            hx = self.hx_fc(hx)
            d_xy = self.mlp(hx).reshape(-1, 1, self.output_size) #control v4
            hx = hx.reshape(1, -1, self.hidden_size)
            # print("dxy", d_xy[0])
            xy = xy + d_xy
            # print("xy plused", xy[0])
            out_wp.append(xy)
        pred_wp = torch.stack(out_wp, dim=1).squeeze(2)
        return pred_wp
    

# time_feature = ["countDown"]
# dynamic_real_feature = [ "SPEED", "HEADING", "MODE", "turn", "acceleration",
#        'current', 'rain', 'snowfall', 'wind_force', 'wind_direc',
#         "resist_ratio","change_x_factor", "change_y_factor"]# 
# static_categorical_feature = ["is_weekday", 'direction',"season", "departure_hour"] # ScheduleType #"adversarial"
# y_cols = ["FC","SOG","LONGITUDE","LATITUDE"]
# # y_cols = ["FC","SOG"]

# context_length = 24
# prediction_horizon = 5


time_feature = config.time_feature
dynamic_real_feature = config.dynamic_real_feature
static_categorical_feature = config.static_categorical_feature
y_cols = config.y_cols

context_length = config.context_length
prediction_horizon = config.prediction_horizon


config = InformerConfig.from_pretrained("huggingface/informer-tourism-monthly", prediction_length=prediction_horizon,
        context_length=context_length, input_size=len(y_cols), num_time_features=len(time_feature),
        num_dynamic_real_features = len(dynamic_real_feature), num_static_real_features = len(static_categorical_feature),
        lags_sequence=[1], num_static_categorical_features=0, feature_size=31)
# tf_model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly",
#                                                            config=config, ignore_mismatched_sizes=True).to(device)
tf_model = InformerModel.from_pretrained("huggingface/informer-tourism-monthly",
                                        config=config, ignore_mismatched_sizes=True).to(device)

# gru_model = model.GRU_update(device, input_size=len(y_cols), hidden_size=20, output_size=len(y_cols), num_layers=1, prediction_horizon=prediction_horizon).to(device)
#30, 350
gru_model = GRU_update(4, hidden_size=375, output_size=4, num_layers=1, prediction_horizon=5, device=device).to(device)
# model = model.vesselModel(tf_model, gru_model, device, sequence_length, prediction_horizon, context_length, batch_size, y_cols, time_feature, dynamic_real_feature, static_categorical_feature)

tf_model = tf_model.float()



# combine tf_model and GRU_update
class vesselModel(nn.Module):
    def __init__(self, tf_model = tf_model , gru_model = gru_model, output_features = 4):
        super().__init__()
        # self.tf_model = CustomInformer(tf_model, output_features=4)
        self.tf_model = tf_model
        self.fc = nn.Linear(in_features=tf_model.config.hidden_size, out_features=output_features, bias=True).to(device)
        self.gru_model = gru_model
        # self.device = device

    def forward(self, past_values, past_time_features, 
                static_real_features, past_observed_mask, 
                future_time_features, future_values, 
                ):

        tf_output = self.tf_model(past_values=past_values, 
                                past_time_features=past_time_features, 
                                static_real_features=static_real_features,
                                past_observed_mask=past_observed_mask, 
                                future_values=future_values, 
                                future_time_features=future_time_features, 
                                output_hidden_states=True,
                                return_dict=True)
        pred_tf = self.fc(tf_output.last_hidden_state)
        pred_wp = self.gru_model(pred_tf, past_time_features)

        return pred_tf, pred_wp
    

class WeightedMSELoss(nn.Module):
    def __init__(self, weights, device = device):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.device = device

    def forward(self, input, target):
        self.weights = self.weights.to(self.device)
        squared_diff = (input - target) ** 2
        weighted_squared_diff = squared_diff * self.weights
        loss = weighted_squared_diff.mean()

        return loss
