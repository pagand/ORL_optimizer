
import os

class VesselConfig:
    
    
    iter = 5
    n_epochs = 100
    loss_weight = [3,3,1,1] # weight for each output
    # loss_weight_m = [1,1] # weight for tf_loss and gru_loss
    
    # log_wandb = True
    log_wandb = False
    

    batch_size=128
    sequence_length = 25
    context_length = 24
    prediction_horizon = 5 #10

    # initialize the model
    time_feature = ["Time2"]
    dynamic_real_feature = [ "SPEED", "HEADING", "MODE", "turn", "acceleration",
        'current', 'rain', 'snowfall', 'wind_force', 'wind_direc',
            "resist_ratio","change_x_factor", "change_y_factor", "countDown"]# 
    static_categorical_feature = ["is_weekday", 'direction',"season", "departure_hour"] # ScheduleType #"adversarial"
    y_cols = ["FC","SOG","LONGITUDE","LATITUDE"]
    # y_cols = ["FC","SOG"]

    data_len = 90
    

    

# Iter 3
# using pytorch MSE loss
# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr = 1e-6)

# Iter 4
# using custom MseLoss
# weights = [3,2,1,1]


# time_feature = ["countDown"]
# dynamic_real_feature = [ "SPEED", "HEADING", "MODE", "turn", "acceleration",
#     'current', 'rain', 'snowfall', 'wind_force', 'wind_direc',
#         "resist_ratio","change_x_factor", "change_y_factor"]# 
# static_categorical_feature = ["is_weekday", 'direction',"season", "departure_hour"] # ScheduleType #"adversarial"
# y_cols = ["FC","SOG","LONGITUDE","LATITUDE"]


#Iter 5
# using MSE
# dynamic_real_feature = [ "SPEED", "HEADING", "MODE", "turn",                   #"acceleration",
#         'current', 'rain', 'snowfall', 'wind_force', 'wind_direc',
#             "resist_ratio","change_x_factor", "change_y_factor", "countDown"]# 


#Iter 6
#  loss_weight = [4,4,1,1]
# dynamic_real_feature = [ "SPEED", "HEADING", "MODE", "turn", "acceleration",
#         'current', 'rain', 'snowfall', 'wind_force', 'wind_direc',
#             "resist_ratio","change_x_factor", "change_y_factor", "countDown"]# (acceleration calculation wrong)


#Iter 7
# loss_weight = [4,4,1,1]
# dynamic_real_feature = [ "SPEED", "HEADING", "MODE", "turn", "acceleration",
#         'current', 'rain', 'snowfall', 'wind_force', 'wind_direc',
#             "resist_ratio","change_x_factor", "change_y_factor", "countDown"]#

#Iter 8
# loss_weight = [3,3,1,1]
# dropout = 0.1
# epoch = 100
# batch_size = 256



#TF Iter 1
# epoch = 100
# batch_size = 128


# TF Iter 2
# epoch = 100
# batch_size = 64
# loss_weight = [3,3,1,1]




# TF Iter 3
# epoch = 100
# batch_size = 128
# using MSELOSS
# loss_weight = [3,3,1,1]   
# dropout = 0.1
# train with code(output != None )
# 31 features
# Scheduler, step(mean(mse))
# best: 58, 79, 93
# [0] + pred_h/120


# TF Iter 4
# batch_size = 128
# each epoch trained with the min length of autopilot data
# [0] + pred_h
# 49? 71..


# TF Iter 5
# each epoch trained with the min length of autopilot data
# [0] + pred_h/120
# 44 


