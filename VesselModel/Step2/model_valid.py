
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
from arModel import get_testTripId,evaluate_model
import wandb



class GRU_update(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=4, num_layers=1, prediction_horizon=5, device="cpu"):
        super().__init__()
        self.device = device
        self.h = prediction_horizon
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.mlp = nn.Sequential( nn.ReLU(),
                                  nn.Linear(hidden_size, 2048),
                                  nn.Dropout(0.3),
                                  nn.ReLU(),
                                  nn.Linear(2048, output_size))
        self.hx_fc = nn.Linear(2*hidden_size, hidden_size)

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
            # print("dxy", d_xy)
            xy = xy + d_xy
            # print("xy plused", xy)
            out_wp.append(xy)
        pred_wp = torch.stack(out_wp, dim=1).squeeze(2)
        return pred_wp



class testTrip():
    def __init__(self, data, models_file_path):
        # self.tripId = tripId
        # self.df = df
        self.data = data
        self.max_steps = len(self.data) -10
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        # load forecasting models
        self._load_model(models_file_path)
        self._set_eval()

    def _load_model(self,filepath):
        self.time_feature = ["countDown"]
        self.dynamic_real_feature = [ "SPEED", "HEADING", "MODE", "turn", "acceleration",
                                'current', 'rain', 'snowfall', 'wind_force', 'wind_direc',
                                "resist_ratio","change_x_factor", "change_y_factor"]# 
        self.static_categorical_feature = ["is_weekday", 'direction',"season", "departure_hour"] # ScheduleType,"adversarial"
        self.y_cols = ["FC","SOG","LONGITUDE","LATITUDE"]
        config = InformerConfig.from_pretrained("huggingface/informer-tourism-monthly", prediction_length=5,
            context_length=24, input_size=4, num_time_features=1,
            num_dynamic_real_features = 13, num_static_real_features = 4,
            lags_sequence=[1], num_static_categorical_features=0, feature_size=30)
        self.tf = InformerForPrediction(config).to(self.device)
        self.tf.load_state_dict(torch.load(filepath[0], map_location=torch.device(self.device)))

        self.gru = GRU_update(4, hidden_size=350, output_size=4, num_layers=1, prediction_horizon=5, device=self.device).to(self.device)
        self.gru.load_state_dict(torch.load(filepath[1], map_location=torch.device(self.device)))

    def _set_eval(self):
        self.tf.eval()
        self.gru.eval()

        self.current_step = starting_step

    def _step(self):
        self.past_feature = self.data[self.current_step-25:self.current_step][self.time_feature + self.dynamic_real_feature].copy()
        if (self.current_step == starting_step):
            self.past_values = self.data[self.current_step-25:self.current_step][self.y_cols].copy()
            self.static_values = self.data.iloc[0][self.static_categorical_feature].copy()
            # print(self.static_values)
            # self.static_real_features.astype(float)
            # print(self.static_values)
            
        self.future_feature = self.past_feature[-5:].copy()
        
        self.future_feature[self.future_feature.columns[0]] = self.future_feature[self.future_feature.columns[0]] + 5/120

        # predict
        fc, sog, longitude, latitude = self._predict(self.past_values, self.past_feature, self.future_feature, self.static_values,self.tf, self.gru)
        # update past values for next step
        self.past_values = self.past_values[1:self.current_step]
        self.past_values.append(pd.Series(), ignore_index=True)
        self.past_values.loc[self.current_step] = fc, sog, longitude, latitude
        
        return fc, sog, longitude, latitude

    def _predict(self, past_values, past_feature, future_feature, static_feature, tf_model, gru_model):
        future_feature = torch.from_numpy(np.expand_dims(future_feature, 0)).float().to(self.device)
        past_values = torch.from_numpy(np.expand_dims(past_values, 0)).float().to(self.device)
        past_feature = torch.from_numpy(np.expand_dims(past_feature, 0)).float().to(self.device)
        static_feature = torch.from_numpy(np.expand_dims(np.array(static_feature).astype(float), 0)).float().to(self.device)
        past_observed_mask = torch.ones(past_values.shape).to(self.device)
        
        with torch.no_grad():
            outputs = tf_model.generate(past_values=past_values, past_time_features=past_feature,
                    static_real_features=static_feature, past_observed_mask=past_observed_mask,
                    future_time_features=future_feature).sequences.mean(dim=1)
            outputs = gru_model(outputs, past_feature).detach().cpu().numpy()
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
            if self.current_step == self.max_steps:
                break
        print("Trip finished")
        return self.pred_fc, self.pred_sog, self.pred_longitude, self.pred_latitude
    


def plot_trip(data, fc, sog, longitude, latitude):
    fig = plt.figure()
    grid = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)
    plt.title(f'iter_{load_iter} epoch_{load_cp} trip_{tripId}')

    ax0 = plt.subplot(grid[0, 0])
    ax1 = plt.subplot(grid[0, 1:])
    ax2 = plt.subplot(grid[1, :1])
    ax3 = plt.subplot(grid[1, 1:])

    stop = len(fc) + step
    # Plot 'fc'
    ax0.plot(data[step:stop]['countDown'], data[step:stop]['FC'], label='Actual'.format(i=1), color="Green")
    ax0.plot(data[step:stop]['countDown'], fc[:stop], label='Pred AR'.format(i=2), linestyle='dashed', color = "Red")
    ax0.set_xlim(data['countDown'].iloc[step], data['countDown'].iloc[-1])


    ax1.plot(data[step:stop]['countDown'], data[step:stop]['SOG'], label='Actual'.format(i=1), color="Green")
    ax1.plot(data[step:stop]['countDown'], sog[:stop], label='Pred AR'.format(i=2), linestyle='dashed', color = "Red")
    ax1.set_xlim(data['countDown'].iloc[step], data['countDown'].iloc[-1])

    ax2.plot(data[step:stop]['countDown'], data[step:stop]['LONGITUDE'], label='Actual'.format(i=1), color="Green")
    ax2.plot(data[step:stop]['countDown'], longitude[:stop], label='Pred AR'.format(i=2), linestyle='dashed', color = "Red")
    ax2.set_xlim(data['countDown'].iloc[step], data['countDown'].iloc[-1])

    ax3.plot(data[step:stop]['countDown'], data[step:stop]['LATITUDE'], label='Actual'.format(i=1), color="Green")
    ax3.plot(data[step:stop]['countDown'], latitude[:stop], label='Pred AR'.format(i=2), linestyle='dashed', color = "Red")
    ax3.set_xlim(data['countDown'].iloc[step], data['countDown'].iloc[-1])



    # ax0.set_title('Fuel Consumption (fc)')
    ax0.set_xlabel('countDown')
    ax0.set_ylabel('fc')
    # ax1.set_title('Speed Over Ground (sog)')
    ax1.set_xlabel('countDown')
    ax1.set_ylabel('sog')
    # ax2.set_title('Longitude')
    ax2.set_xlabel('countDown')
    ax2.set_ylabel('Longitude')
    # ax3.set_title('Latitude')
    ax3.set_xlabel('countDown')
    ax3.set_ylabel('Latitude')

    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax0.set_ylim(0,1)
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,1)  
    ax3.set_ylim(0,1)

    plot_filename = f'iter_{load_iter} epoch_{load_cp} trip_{tripId}, .png'

    plt.savefig("Plot/{}".format(plot_filename))
    wandb.log({f'iter_{load_iter} epoch_{load_cp}': wandb.Image("Plot/{}".format(plot_filename))})



testTripsIDs = get_testTripId()
print(testTripsIDs)

path = "data/Features/feature4.csv"
df = pd.read_csv(path)

# df.iloc[0, df.columns.get_loc('prev_HEADING')] = df.iloc[1, df.columns.get_loc('prev_HEADING')]
# df.iloc[0, df.columns.get_loc('prev_SOG')] = df.iloc[1, df.columns.get_loc('prev_SOG')]
df.iloc[0, df.columns.get_loc('turn')] = df.iloc[1, df.columns.get_loc('turn')]
# df.iloc[0, df.columns.get_loc('acceleration')] = 0

starting_step = 25
load_iter = 15
load_cp = 54
model_name = "Model_Iter_{}".format(load_iter)

filepath = ("data/Checkpoints/{}/{}_checkpoint{}.pt".format(model_name, model_name,load_cp),
            "data/Checkpoints/{}/{}_checkpoint{}_gru.pt".format(model_name,model_name,load_cp))

random.seed(None)


wandb.login()
wandb.init(project="visualization", name="gru_{}_checkpoint{}".format(load_iter, load_cp))

tripIds = random.sample(testTripsIDs, 5)

for tripId in tripIds:
    data = df[df["trip_id"]==tripId].reset_index(drop=True)

    print("Selected trip: ", tripId, "Is Adversaril trip: ", data.iloc[0].adversarial)
    testTrips = testTrip(data, filepath)
    fc, sog, longitude, latitude = testTrips.run()

    step = starting_step
    stop = -1
    print(len(data[step:stop]), len(fc), len(sog),len(longitude), len(latitude))

    plot_trip(data, fc, sog, longitude, latitude)

