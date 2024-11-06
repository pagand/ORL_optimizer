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
from model import GRU_update, WeightedMSELoss
import data
from config import VesselConfig as config


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


batch_size=config.batch_size
sequence_length = config.sequence_length
context_length = config.context_length
prediction_horizon = config.prediction_horizon

time_feature = config.time_feature
dynamic_real_feature = config.dynamic_real_feature
static_categorical_feature = config.static_categorical_feature
y_cols = config.y_cols


tfconfig = InformerConfig.from_pretrained("huggingface/informer-tourism-monthly", prediction_length=prediction_horizon,
        context_length=context_length, input_size=len(y_cols), num_time_features=len(time_feature),
        num_dynamic_real_features = len(dynamic_real_feature), num_static_real_features = len(static_categorical_feature),
        lags_sequence=[1], num_static_categorical_features=0, feature_size=31)

tf_model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly",
                                        config=tfconfig, ignore_mismatched_sizes=True).to(device)

gru_model = GRU_update(4, hidden_size=375, output_size=4, num_layers=1, prediction_horizon=5, device=device).to(device)


tf_model = tf_model.float()

def train(trainloader, validloader, model_name):
    best_loss = 10000
    best_epoch = -1

    for epoch in range(config.n_epochs):
        epoch_tf_loss = []
        epoch_gru_loss = []
        tf_model.train()
        gru_model.train()

        for _, data in enumerate(trainloader):
            batch_loss = 0
            batch_tf_loss = []
            batch_gru_loss = []
            predicted = None
            values, time_features, static_real_features, future_features, actions, done = data
            # print(done.shape)
            # print(done)
            earliest = torch.min(done)
            values_tmp = torch.clone(values)
            # for t in range(sequence_length, values.shape[1]-prediction_horizon):  
            for t in range(sequence_length, earliest-prediction_horizon):     
                if t!=25:
                    predicted = predicted.detach()
                    values_tmp[:, t-1, :] =  predicted[:, 0, :]
                    acceleration = predicted[:,0,1]- values_tmp[:, t-2, 1]
                    time_features[:, t-1, 5] = acceleration
                future_time_features = time_features[:, t-prediction_horizon:t]
                future_time_features[:, 0, 1:5] = time_features[:, t, 1:5]  # from speed to turn
                future_time_features[:, :, 0] = future_time_features[:, :, 0]+prediction_horizon/120
                future_time_features[:,:, -1] = future_time_features[:,:, -1] - 1
                
                future_values = values[:, t:t+prediction_horizon]
                past_values = values[:, t-sequence_length: t]
                past_time_features = time_features[:, t-sequence_length: t]

                

                past_observed_mask = torch.ones_like(past_values).to(device)
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                tf_out = tf_model(past_values=past_values, past_time_features=past_time_features, static_real_features=static_real_features,
                            past_observed_mask=past_observed_mask, future_values=future_values, future_time_features=future_time_features)
                tf_loss = tf_out.loss

                tf_loss.backward()

                with torch.no_grad():
                    output = tf_model.generate(past_values=past_values, past_time_features=past_time_features, static_real_features=static_real_features,
                                           past_observed_mask=past_observed_mask, future_time_features=future_time_features).sequences.mean(dim=1) 
                predicted = gru_model(output, past_time_features)

                gru_loss = criterion(predicted, future_values) 
                gru_loss.backward()

                optimizer1.step()
                optimizer2.step()

                batch_tf_loss.append(tf_loss.item())
                batch_gru_loss.append(gru_loss.item())

            batch_tf_loss = np.mean(batch_tf_loss)
            batch_gru_loss = np.mean(batch_gru_loss)

            epoch_tf_loss.append(batch_tf_loss)
            epoch_gru_loss.append(batch_gru_loss)
        
        epoch_tf_loss = np.mean(epoch_tf_loss)
        epoch_gru_loss = np.mean(epoch_gru_loss)
        
        valid_gru_loss, mses, r2s = evaluate(validloader)
        avg_mse = np.mean(mses)
        scheduler1.step(avg_mse)
        scheduler2.step(avg_mse)

        print("-------- Epoch %d / %d "% (epoch, config.n_epochs))
        print("-------- epoch_loss: ", epoch_gru_loss)
        print("-------- valid_loss: ", valid_gru_loss)
        print("-------- valid_r2: ", r2s)
        print("-------- valid_mses: ", mses)

        if (valid_gru_loss < best_loss):
            best_loss = valid_gru_loss
            best_epoch = epoch
            print("best epoch: ", best_epoch, "best loss: ", best_loss)
            path = "data/Checkpoints/{}/best_tf.pt".format(model_name)
            torch.save(tf_model.state_dict(), path)
            path = "data/Checkpoints/{}/best_gru.pt".format(model_name)
            torch.save(gru_model.state_dict(), path)
        

        dict = {"epoch": epoch, 
                # "epoch_tf_loss": epoch_tf_loss, 
                "epoch_gru_loss": epoch_gru_loss, 
                # "epoch_loss": epoch_tf_loss+epoch_gru_loss,
                # "valid_tf_loss": valid_tf_loss, 
                "valid_gru_loss": valid_gru_loss, 
                # "valid_loss": valid_gru_loss,
                "valid_r2s_FC": r2s[0],
                "valid_r2s_SOG": r2s[1],
                "valid_r2s_LONGITUDE": r2s[2],
                "valid_r2s_LATITUDE": r2s[3], 
                "valid_mses_FC": mses[0],
                "valid_mses_SOG": mses[1],
                "valid_mses_LONGITUDE": mses[2],
                "valid_mses_LATITUDE": mses[3]}
        if(config.log_wandb):
            wandb.log(dict)
        

        
        # path = "data/Checkpoints/{}/{}_epoch_{}_tf.pt".format(model_name, model_name, epoch)
        # torch.save(tf_model.state_dict(), path)
        # path = "data/Checkpoints/{}/{}_epoch_{}_gru.pt".format(model_name, model_name, epoch)
        # torch.save(gru_model.state_dict(), path)
    
    # save last model
    path = "data/Checkpoints/{}/last_tf.pt".format(model_name)
    torch.save(tf_model.state_dict(), path)
    path = "data/Checkpoints/{}/last_gru.pt".format(model_name)
    torch.save(gru_model.state_dict(), path)
    return best_epoch


def evaluate(validloader):
    valid_tf_loss = []
    valid_gru_loss = []
   
    tf_model.eval()
    gru_model.eval()

    r2s = [[] for _ in range(len(y_cols))]
    mses = [[] for _ in range(len(y_cols))]
    residuals = [[] for _ in range(len(y_cols))]
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(validloader):
            predicted = None
            act = []
            pred = []
            values, time_features, static_real_features, future_features, actions, done = data
            earliest = torch.min(done)
            values_tmp = torch.clone(values)
            # for t in range(sequence_length, values.shape[1]-prediction_horizon):   
            for t in range(sequence_length, earliest-prediction_horizon):    
                if t!=25:
                    predicted = predicted.detach()
                    values_tmp[:, t-1, :] =  predicted[:, 0, :]
                    acceleration = predicted[:,0,1]- values_tmp[:, t-2, 1]
                    time_features[:, t-1, 5] = acceleration
                future_time_features = time_features[:, t-prediction_horizon:t]
                future_time_features[:, 0, 1:5] = time_features[:, t, 1:5]  # from speed to turn
                future_time_features[:, :, 0] = future_time_features[:, :, 0]+prediction_horizon/120
                future_time_features[:,:, -1] = future_time_features[:,:, -1] - 1
                
                future_values = values[:, t:t+prediction_horizon]
                past_values = values_tmp[:, t-sequence_length: t]
                past_time_features = time_features[:, t-sequence_length: t]

                # past_observed_mask = torch.ones_like(past_values).to(device)

                past_observed_mask = torch.ones_like(past_values).to(device)

                
                # tf_out = tf_model(past_values=past_values, past_time_features=past_time_features, static_real_features=static_real_features,
                #         past_observed_mask=past_observed_mask, future_values=future_values, future_time_features=future_time_features)
                # tf_loss = tf_out.loss
                output = tf_model.generate(past_values=past_values, past_time_features=past_time_features, static_real_features=static_real_features,
                                        past_observed_mask=past_observed_mask, future_time_features=future_time_features).sequences.mean(dim=1) 
                predicted = gru_model(output, past_time_features)

                gru_loss = criterion(predicted, future_values) 

                yhat = predicted[:, 0, :].detach().cpu().numpy()
                actual = values[:, t, :].detach().cpu().numpy()

                # print(yhat, actual)

                act.append(actual)
                pred.append(yhat)

                # valid_tf_loss.append(tf_loss.item())
                valid_gru_loss.append(gru_loss.item())

                
        # valid_tf_loss = np.mean(valid_tf_loss)
        valid_gru_loss = np.mean(valid_gru_loss)

        act = np.stack(act, axis=1)
        pred = np.stack(pred, axis=1)


        for i in range(len(y_cols)):
            mse = mean_squared_error(act[:, i], pred[:, i])
            mses[i].append(mse)
            # squared_diff = (act[:,:,i] - pred[:,:,i]) ** 2
            # weighted_squared_diff = squared_diff * config.loss_weight[i]
            # weighted_mse = np.mean(weighted_squared_diff)
            # mses[i].append(weighted_mse)
            actual_1 = act[:, i].swapaxes(1, 0).reshape([-1, 1])
            yhat_1 = pred[:, i].swapaxes(1, 0).reshape([-1, 1])
            r2 = r2_score(actual_1, yhat_1)
            # print("actual_l shape: ", actual_1.shape, "actual_1: ", actual_1)
            r2s[i].append(r2)
            # print(r2)
            # exit()

        predictions.append(pred)
        actuals.append(act)
        
    r2s = [sum(x)/len(x) for x in r2s]
    mses = [sum(x)/len(x) for x in mses]

    return valid_gru_loss, mses, r2s
         

trainset = data.vessel_data(train = True, train_test_split = 0.8, rand_seed=2)
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, drop_last=True)

validset = data.vessel_data(train = False, train_test_split = 0.8, rand_seed=2)
validloader = DataLoader(validset, batch_size = batch_size, drop_last=True)

testset = data.vessel_data(train = False, test=True, train_test_split = 0.8, rand_seed=2)
testloader = DataLoader(testset, batch_size = batch_size, drop_last=True)

optimizer1 = optim.Adam(tf_model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min', patience=3, factor=0.5, min_lr = 1e-6, verbose=True)

optimizer2 = optim.Adam(gru_model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', patience=3, factor=0.5, min_lr = 1e-6, verbose=True)

criterion = nn.MSELoss()
# criterion = WeightedMSELoss(config.loss_weight)



n_epochs = config.n_epochs
model_name = "Model_{}_Iter_{}".format(config.version, config.iter)
# best_epoch = 95

if not os.path.exists("data/Checkpoints/{}".format(model_name)):
    os.makedirs("data/Checkpoints/{}".format(model_name))

# if not os.path.exists("data/Checkpoints/Best"):
#     os.makedirs("data/Checkpoints/Best")


print("start training iteration: ", iter)

if(config.log_wandb):
    wandb.login()
    wandb.init(project="VesselModel", name="{}".format(model_name))
    wandb.watch(tf_model)
    wandb.watch(gru_model)

# best_epoch = 93
best_epoch = train(trainloader, validloader, model_name)
config.best_epoch = best_epoch


#test
tf_model.load_state_dict(torch.load("data/Checkpoints/{}/best_tf.pt".format(model_name)))
gru_model.load_state_dict(torch.load("data/Checkpoints/{}/best_gru.pt".format(model_name)))

print("----- Test Result ------ ")
print("best epoch: ", best_epoch)
test_gru_loss, mses, r2s = evaluate(testloader)

print("test_loss: ", test_gru_loss)
print("test_r2: ", r2s)
print("test_mses: ", mses)
