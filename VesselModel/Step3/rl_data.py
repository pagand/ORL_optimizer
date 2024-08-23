import pandas as pd
import numpy as np
import pickle
import inspect

df = pd.read_csv("data/Features/feature3.csv")
# df.columns

# # H-N
# d1 = df[df.direction==1][["LONGITUDE", "LATITUDE","FC", "trip_id"]] # h-n
# d2 = df[df.direction==0][["LONGITUDE", "LATITUDE","FC", "trip_id"]] # n-h

# # get the avg location of the top 1 percent trip
# def get_top1_location(d1):
#     total = d1.trip_id.nunique() // 100
#     top_1_trips = list(d1.groupby("trip_id").FC.sum().sort_values()[:total].index)
#     i = top_1_trips.pop(0)
#     hn_top = d1[d1.trip_id==i].reset_index(drop=True)
#     if (len(hn_top) < 200):
#         for j in range(200-len(hn_top)):
#             hn_top = hn_top.append(hn_top.tail(1)).reset_index(drop=True)
#     for i in top_1_trips:
#         tmp_df2 = df[df.trip_id==i].reset_index(drop=True)
#         if (len(tmp_df2) < 200):
#             for j in range(200-len(tmp_df2)):
#                 tmp_df2 = tmp_df2.append(tmp_df2.tail(1)).reset_index(drop=True)
#         hn_top["LONGITUDE"] = hn_top.LONGITUDE + tmp_df2.LONGITUDE
#         hn_top["LATITUDE"] = hn_top.LATITUDE + tmp_df2.LATITUDE
#     hn_top["LONGITUDE"] = hn_top["LONGITUDE"]/(len(top_1_trips)+1)
#     hn_top["LATITUDE"] = hn_top["LATITUDE"]/(len(top_1_trips)+1)
#     return hn_top[["LONGITUDE", "LATITUDE"]]


# hn_top = get_top1_location(d1)
# nh_top = get_top1_location(d2)

# hn_top.to_csv("data/Features/H2N_top1.csv", index=False)
# nh_top.to_csv("data/Features/N2H_top1.csv", index=False)

hn_top = pd.read_csv("data/Features/H2N_top1.csv")
nh_top = pd.read_csv("data/Features/N2H_top1.csv")


time_feature = ["Time2"]
dynamic_feature = [ 
                   "turn", "acceleration",'current', 'rain',
                     'snowfall', 'wind_force', 'wind_direc',"resist_ratio",
                     "change_x_factor", "change_y_factor", "countDown"]# 
static_categorical_feature = ["is_weekday", 'direction',"season", "departure_hour"] # ScheduleType #"adversarial"
y_cols = ["FC","SOG","LONGITUDE","LATITUDE"]
actions = ["SPEED", "HEADING", "MODE"]

df["goal_long"] = df.direction.apply(lambda x: 0.9965111208024382 if x==1 else 0.0023259194650222526)
df["goal_lat"] = df.direction.apply(lambda x: 0.7729570345408661 if x==1 else 0)

def get_observation (df, feature_cols = time_feature + dynamic_feature + static_categorical_feature + y_cols,
                 action_cols = actions):
    features = df.copy()   
    rewards_col = ["trip_id", "LONGITUDE", "LATITUDE", "direction", "goal_long", "goal_lat","countDown"]
    rewards_df = features[rewards_col]    
    # apply minmax scaler for fuel consumption to get reward 2    
    rewards_df["reward2"] = - features[["FC"]]    
    dataset_list = []    
    for i in list(df.trip_id.unique()):              
        data_dict = {}        
        # observations        
        observation = features[features.trip_id==i].drop("trip_id", axis=1)[feature_cols]
        observation = observation.to_numpy().astype(float)        
        data_dict["observations"] = observation       
        
         # next_observations\n        
        observation = np.delete(observation, 0, 0)        
        last = observation[-1]        
        observation = np.vstack([observation, last])        
        data_dict["next_observations"] = observation        
        
        # actions       
        # # print(i)      
        actions = features[features.trip_id==i].drop("trip_id", axis=1)[action_cols]       
        # actions = actions.drop("trip_id", axis=1)        
        data_dict["actions"] = actions.to_numpy().astype(float)        
        
        # rewards       
        rewards = rewards_df[rewards_df.trip_id==i].reset_index()        
        rewards        
        
        # reward1 distance to top 1\n        
        trip_len = rewards.shape[0]        
        if rewards.loc[0,"direction"]==1:            
            top1 = hn_top.iloc[:trip_len]        
        else:
            top1 = nh_top.iloc[:trip_len]        
        rewards["reward1"] = - ((rewards["LONGITUDE"]-top1["LONGITUDE"])**2 + \
                                (rewards["LATITUDE"]-top1["LATITUDE"])**2 )**0.5        
        # rewards["reward1"]
        rewards["reward1"] = rewards["reward1"].apply(lambda x: 0 if x > -0.05 else x)        
        rewards["reward1"] = rewards["reward1"]
        # reward2 fc consumption and done reward       
        rewards.loc[len(rewards)-1,"reward2"] = rewards.loc[len(rewards)-1,"reward2"]+3        
        
        # rewards4 time out reward        
        # rewards["Time"] = rewards.index        
        # rewards.reset_index(inplace=True)
        # rewards["reward3"] = rewards["Time"].apply(lambda x: -0.1*((x-90)//10) if x > 100 else 0)
        rewards["reward3"] = rewards["countDown"].apply(lambda x: 0.1*x if x < 0 else 0)
        
        data_dict["rewards"] = (rewards[["reward1","reward2", "reward3"]]).to_numpy()
         # termination      
        termination = np.zeros([observation.shape[0],1])
        termination[-1,0] = 1
        data_dict["termination"] = termination
        
        dataset_list.append(data_dict)
        
    return dataset_list



rl_data = get_observation(df)
with open('data/rl_data.pkl', 'wb') as handle:
    pickle.dump(rl_data,handle)


# df[df["trip_id"] ==14][["Time2", "turn", "acceleration", "current", "rain", "snowfall", 
#                                    "wind_force", "wind_direc", "resist_ratio", "change_x_factor", "change_y_factor", "countDown",
#                                      "is_weekday", "direction", "season", "departure_hour", "FC", "SOG", "LONGITUDE", "LATITUDE"]].iloc[25]