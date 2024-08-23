import os
import sys
import pickle
import random
import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from transformers import InformerForPrediction, InformerConfig
# import tkinter as tk
# from tkinter import *
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Step2 import model
from Step2.config import VesselConfig as config
from Step1 import convention

class VesselEnvironment(gym.Env):
    """An environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    # model = [tf_loc, gru_loc, tf_fc, gru_fc]
    def __init__(self, rl_data, scaler, toptrips, models_file_path, reward_type = "mimic"):
        self.rl_data = rl_data
        self.manual = False
        self.run = False
        self.done =False
        self.scale_var = False
        self.max_steps = 124
        self.trip_id = 0
        self.reward_type = reward_type
        # load best 1% trips to calculate reward1
        self.hn_top = toptrips[0]
        self.nh_top = toptrips[1]
        # set scaler
        self.scaler = scaler
        # get device
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        # load forecasting models
        self._load_model(models_file_path)
        self._set_eval()
    

    def _load_model(self, filepath):
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
    
    def _set_eval(self):
        self.gru.eval()
        self.tf.eval()

        # initialize values
        self.current_step = 25
        self.reward_cum = 0
        self.reward = 0
        self.obs = np.zeros([1,19], dtype=np.float64)
        self.actions = np.zeros([1,3], dtype=np.float64)

    def _get_observation(self):
        # print("get_obs", self.obs[-25])
        return self.obs[-25:]
    
    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if(self.trip_id < len(self.rl_data)-1):
            self.trip_id = self.trip_id + 1
        else:
            self.trip_id = 1

        self.data = self.rl_data[self.trip_id]["observations"]
        self.current_step = 25
        self.obs = self.rl_data[self.trip_id]["observations"][0:25]
        self.actions = self.rl_data[self.trip_id]["actions"][0:25]

        # get direction and other static features
        self.direction = self.rl_data[self.trip_id]["observations"][0, 13]
        self.statics = self.rl_data[self.trip_id]["observations"][0, 12:16]
        if self.direction==1:
            self.top1 = self.hn_top
            self.goal_long, self.goal_lat = np.float64(0.9965111208024382), np.float64(0.7729570345408661)
        else:
            self.top1 = self.nh_top
            self.goal_long, self.goal_lat = np.float64(0.0023259194650222526), np.float64(0)

        # calculate the cumulative reward
        self.reward_cum = 0

        return self._get_observation(), {}
    

    # obs_cols = [0"Time2", 1"turn", 2"acceleration",
    #    3'current', 4'rain', 5'snowfall',6 'wind_force',7 'wind_direc', 8"resist_ratio",
    #   '9change_x_factor', 10 'change_y_factor', 11 'countDown',
    #    12"is_weekday",13 'direction', 14"season",15"hour", 
    #    16"FC", 17"SOG", 18"LONGITUDE", 19'LATITUDE',
    #    ], 
    # action_cols = ["SPEED", "HEADING", "MODE"]
    # time_feature = ["Time2"]
    # dynamic_real_feature = [ 1 "SPEED", 2 "HEADING", 3 "MODE", 4 "turn",  5"acceleration",
    #    6 'current',7 'rain', 8'snowfall', 9'wind_force', 10'wind_direc',
    #       11  "resist_ratio",12 "change_x_factor",13  "change_y_factor", 14 "countDown"]#14
    # static_categorical_feature = ["is_weekday", 'direction',"season", "departure_hour"] # ScheduleType #"adversarial"
    # y_cols = ["FC","SOG","LONGITUDE","LATITUDE"]

    def _take_action(self, action):
        # get actions
        speed, heading, mode = action
        mode = int(mode>0.5)
        if self.current_step < self.rl_data[self.trip_id]["observations"].shape[0]:
            future_obs = self.rl_data[self.trip_id]["observations"][self.current_step].copy()
        else:
            future_obs = self.rl_data[self.trip_id]["observations"][-1].copy()
        obs = self._get_observation().copy()

        actions = self.actions[-25:].copy()

        # print(obs[0], obs[-5], obs[-1])
        # print(actions[0])

        past_time_feature = np.zeros([25, len(self.dynamic_real_feature)+1])
        past_time_feature[:,0] = obs[:,0]
        past_time_feature[:, 1:4] = actions[:] # speed, heading, mode
        past_time_feature[:, 4:15] = obs[:, 1:12]
        
        # print(past_time_feature[0])
        # print(past_time_feature[0])
        # print(self.current_step, future_obs)

        future_time_feature = past_time_feature[-5:].copy()
        future_time_feature[:,0] = future_time_feature[:,0] + 5/120 # Time2
        future_time_feature[0,[1,2,3]] = [speed, heading, mode]
        future_time_feature[0, 4] = heading - past_time_feature[-1, 2] #turn
        # print("heading", heading, past_time_feature[-1, 2])
        change_x = np.cos((heading+90) * np.pi / 180)
        change_y = np.sin((heading-90) * np.pi / 180)
        future_time_feature[0, [12,13]] = change_x, change_y
        future_time_feature[:, 14] = future_time_feature[:, 14] -1 # countDown

        past_values = obs[:, -4:]

        # if (self.current_step < 26):
        #     print("data ", self.rl_data[self.trip_id])
        # print("obs", obs)
        # print("past_time_feature", past_time_feature)
        # print("future_time_feature", future_time_feature)
        # print("past_values", past_values)
        # print("future_obs", future_obs)


        fc, sog, long, lat = self._predict(past_values, past_time_feature, future_time_feature, self.tf, self.gru)
        
        # next observation
        new_obs = future_obs
        # new_obs[0] = future_time_feature[0,0]
        future_time_feature[0,5] = sog - past_time_feature[-1, 5] # acceleration
        new_obs[1:3] = future_time_feature[0, 4:6] #turn, acceleration
        new_obs[-4:] = fc, sog, long, lat

        # print("predicted ", fc, sog, long,lat)
        # print("new_obs", new_obs)

        # update observations and actions
        self.obs = np.append(self.obs, np.expand_dims(new_obs, 0), axis=0)
        self.actions = np.append(self.actions, np.expand_dims(action, 0), axis=0)
        
        return fc, sog, long, lat  
    
    def _predict(self, past_values, past_feature, future_feature, tf_model, gru_model):
        future_feature = torch.from_numpy(np.expand_dims(future_feature, 0)).float().to(self.device)
        past_values = torch.from_numpy(np.expand_dims(past_values, 0)).float().to(self.device)
        past_feature = torch.from_numpy(np.expand_dims(past_feature, 0)).float().to(self.device)
        static_feature = torch.from_numpy(np.expand_dims(self.statics, 0)).float().to(self.device)
        past_observed_mask = torch.ones(past_values.shape).to(self.device)
        
        with torch.no_grad():
            tf_out = tf_model.generate(past_values=past_values, past_time_features=past_feature,
                    static_real_features=static_feature, past_observed_mask=past_observed_mask,
                    future_time_features=future_feature).sequences.mean(dim=1)
            outputs = gru_model(tf_out, past_feature).detach().cpu().numpy()
        return outputs[0,0,0], outputs[0,0,1], outputs[0,0,2], outputs[0,0,3]

    def _get_reward(self, long, lat, fc):
        # reward 1 distance to the top 1
        reward1 = - ((long-self.top1.loc[self.current_step, "LONGITUDE"])**2 + (lat-self.top1.loc[self.current_step, "LATITUDE"])**2 )**0.5
        if reward1 > -0.05:
            reward1 = 0
        reward1 = reward1*10
        # reward 2 fc 
        reward2 = -fc
        
        # reward 3 mimic reward
        if self.current_step < len(self.data):
            reward3 = - ((long-self.data[self.current_step, 18])**2 + (lat-self.data[self.current_step, 19])**2 )**0.5
        else:
            reward3 = - ((long-self.data[-1, 18])**2 + (lat-self.data[-1, 19])**2 )**0.5
        # # reward 4 timeout reward
        # reward4 = 0
        
        countDown = self.data[self.current_step, 11]
        # if self.current_step >= 100:
            # reward4 = -0.1*((self.current_step-90)//10)
       
        # reward4: timeout penalty
        reward4 = 0
        if (countDown < 0): # time out
            reward4 = 0.1*countDown
        
        # reward5: docking reward
        done = (((long-self.goal_long)**2 + (lat-self.goal_lat)**2) < 5e-2)
        if (done) & (countDown >= 0): # arriving docking area
            reward5 = 3
        elif (done) & (countDown < 0): # time out but arrived
            reward5 = 1
        else:
            reward5 = 0
        
        print("reward1: ", reward1, "reward2: ", reward2, "reward3: ", reward3, "reward4: ", reward4, "reward5: ", reward5)
        # return (reward1 + reward2 + reward3 + reward4) / 4

        # if reward1 < -0.05: # arriving docking area
        #     reward5 = 3
        
        if self.reward_type == "mimic":
            self.reward = (reward1 + reward2 + reward3 + reward4 + reward5) / 5
        elif self.reward_type == "top1":
            self.reward = (reward1+reward2+reward4 + reward5) / 4
        else:
            self.reward = (reward2+reward4 + reward5) /3
        
        return self.reward


    def step(self, action):
        obs= self._get_observation()

        fc, sog, long, lat = self._take_action(action) 

        # get done and termination
        done = (((long-self.goal_long)**2 + (lat-self.goal_lat)**2) < 5e-2)
        termination =  self.current_step >= self.max_steps

        reward = self._get_reward(long, lat, fc)
        self.reward_cum = self.reward_cum + reward

        if done:
            # reward = reward+1/3
            # reward = reward + 1
            print("Trip done at step: ", self.current_step)

        self.reward = reward

        print("Step: ", self.current_step, "Reward: ", reward, "Cumulative Reward: ", self.reward_cum, "Done: ", done, "Termination: ", termination)
        print("Action: ", action, "Predicted: ", fc, sog, long, lat)
        
        self.current_step += 1
        return obs, reward, done, termination, {}


def main():
    with open('data/rl_data.pkl', 'rb') as handle:
        rl_data = pickle.load(handle)
    scaler = convention.load_scaler()

    hn_top = pd.read_csv("data/Features/H2N_top1.csv")
    nh_top = pd.read_csv("data/Features/N2H_top1.csv")
    toptrips = (hn_top, nh_top)

    #load model
    # file_path = ["data/Checkpoints/{}/{}_epoch_{}_tf.pt".format(config.model_name, config.model_name, 44), 
    #                     "data/Checkpoints/{}/{}_epoch_{}_gru.pt".format(config.model_name, config.model_name, 44)]
    model_name = "Model_v1_Iter_5"
    file_path = ["data/Checkpoints/{}/{}_epoch_{}_tf.pt".format(model_name, model_name, 44),
                "data/Checkpoints/{}/{}_epoch_{}_gru.pt".format(model_name, model_name, 44)]
    
    print("loading checkpoints: ", file_path[0], file_path[1])
    
    env = VesselEnvironment(rl_data, scaler, toptrips, file_path)
    env.reset()

    fc_predicted = []
    sog_predicted = []
    lat_predicted = []
    long_predicted = []
    trip_ids = []

    # predicted value
    # trip_id = env.trip_id
    repeat = 2
    for i in range(repeat):
        env.reset()
        print("---------- ", env.trip_id, "----------------")
        trip_ids.append(env.trip_id)
        length = rl_data[env.trip_id]["observations"].shape[0]
        fc = np.zeros((length))
        sog =  np.zeros((length))
        lat = np.zeros((length))
        long = np.zeros((length))
        for j in range(25, length):
            action = rl_data[env.trip_id]["actions"][j]
            # action[1] = action[1]
            obs = env.step(action)[0][-1, :]
            # done = env.step(action)[2]
            fc[j], sog[j], long[j], lat[j] = obs[-4], obs[-3], obs[-2], obs[-1]
            # if done:
            #     break
        array1 = np.zeros((length, 12))
        array1[:, 6] = fc
        array1[:, 9] = sog
        array1[:, 7] = lat
        array1[:, 8] = long
        array1 = convention.inverse_transform_value(array1)
        fc_predicted.append(array1[:, 6])
        sog_predicted.append(array1[:, 9])
        lat_predicted.append(array1[:, 7])
        long_predicted.append(array1[:, 8])
        

    # actual value
    fcs = []
    sogs = []
    longs = []
    lats = []
    print(trip_ids)
    for j in range(repeat):
        tripid = trip_ids[j]
        print(tripid)
        array = np.zeros((len(rl_data[tripid]["observations"]), 12))
        array[:, 6] = rl_data[tripid]["observations"][:, -4]
        array[:, 9] = rl_data[tripid]["observations"][:, -3]
        array[:, 8] = rl_data[tripid]["observations"][:, -2]
        array[:, 7] = rl_data[tripid]["observations"][:, -1]
        array = convention.inverse_transform_value(array)
        fcs.append(array[:, 6])
        sogs.append(array[:, 9])
        longs.append(array[:, 8])
        lats.append(array[:, 7])
        
    # array = np.zeros((len(rl_data[env.trip_id]["observations"]), 12))
    # array[:, 6] = rl_data[env.trip_id]["observations"][:, -4]
    # array[:, 9] = rl_data[env.trip_id]["observations"][:, -3]
    # array[:, 7] = rl_data[env.trip_id]["observations"][:, -2]
    # array[:, 8] = rl_data[env.trip_id]["observations"][:, -1]
    # # array = scaler.inverse_transform(array)
    # fcs.append(array[:, 6])
    # sogs.append(array[:, 9])
    # lats.append(array[:, 7])
    # longs.append(array[:, 8])

    def plot(i):
        fig = plt.figure()
        grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)

        ax1 = plt.subplot(grid[0, 0])
        ax2 = plt.subplot(grid[0, 1:])
        ax3 = plt.subplot(grid[1, :1])
        ax4 = plt.subplot(grid[1, 1:])

        # plt.xlim(0,1)
        # plt.ylim(0,1)

        ax1.plot(range(25, len(fc_predicted[i])), fc_predicted[i][25:], label='predictions'.format(i=2))
        ax1.plot(range(25, len(fc_predicted[i])), fcs[i][25:], label='actuals'.format(i=1))
        ax1.legend(loc='best')
        ax2.plot(range(25, len(fc_predicted[i])), long_predicted[i][25:], label='predictions'.format(i=2))
        ax2.plot(range(25, len(fc_predicted[i])), longs[i][25:], label='actuals'.format(i=1))
        ax2.legend(loc='best')
        ax3.plot(range(25, len(fc_predicted[i])), lat_predicted[i][25:], label='predictions'.format(i=2))
        ax3.plot(range(25, len(fc_predicted[i])), lats[i][25:], label='actuals'.format(i=1))
        ax3.legend(loc='best')
        ax4.plot(range(25, len(sog_predicted[i])), sog_predicted[i][25:], label='predictions'.format(i=2))
        ax4.plot(range(25, len(sog_predicted[i])), sogs[i][25:], label='actuals'.format(i=1))
        ax4.legend(loc='best')
        plt.savefig("Plot/gym_plot_{}.png".format(trip_ids[i]))
        plt.show()
    
    for i in range(repeat):
        plot(i) 
    print("done")


if __name__ == "__main__":
    main()