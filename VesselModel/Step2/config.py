
import os

class VesselConfig:
    
    
    iter = 5
    version = "v1"
    ckpt = "best" # "best" or "last"
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

    data_len = 90
    

    
