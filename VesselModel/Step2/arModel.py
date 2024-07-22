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


path = "data/Features/feature4.csv"
df = pd.read_csv(path)

df.columns

df.iloc[0, df.columns.get_loc('prev_HEADING')] = df.iloc[1, df.columns.get_loc('prev_HEADING')]
df.iloc[0, df.columns.get_loc('prev_SOG')] = df.iloc[1, df.columns.get_loc('prev_SOG')]
df.iloc[0, df.columns.get_loc('turn')] = df.iloc[1, df.columns.get_loc('turn')]
df.iloc[0, df.columns.get_loc('acceleration')] = 0


columns = df.columns.drop("countDown")
shifted_df = df.copy()
shifted_df[columns] = df[columns].shift(periods=5)
data_len = 80


class vessel_data(Dataset):
    def __init__(self, train = True, test=False, train_test_split = 0.7, rand_seed=1):
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
        trip_id = (self.trips_id[idx//2])
        # trip_id = self.trips_id[idx]
        # start = self.starting_dict[trip_id]
        # start_idx = self.starting_dict[trip_id] + (idx%3)*(self.prediction_horizon+self.sequence_length)
        data = df[df.trip_id==trip_id].reset_index(drop=True)
        shifted_data = shifted_df[df.trip_id==trip_id].reset_index(drop=True)
        starting = random.randint(0, math.floor((len(data)-data_len)/2))
        # starting = 0
        # if (idx%2):
        #     starting = len(data)-91

        # starting = 0
        data = data.iloc[starting:starting+data_len]

        values = data[y_cols]
        time_features = data[time_feature + dynamic_real_feature]
        static_categorical_features = data.iloc[0][static_categorical_feature]
        future_time_features = shifted_data.iloc[starting:starting+data_len][time_feature + dynamic_real_feature]
        actions = data[["SPEED", "HEADING", "MODE", "turn"]]
        return self.df_to_tensor(values), self.df_to_tensor(time_features), self.df_to_tensor(static_categorical_features), self.df_to_tensor(future_time_features), self.df_to_tensor(actions)

    def __len__(self):
        return len(self.trips_id)*2
        # return len(self.trips_id)


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

batch_size=256
sequence_length = 25
context_length = 24
prediction_horizon = 5 #10
# criterion = Weighted_Loss()

time_feature = ["countDown"]
dynamic_real_feature = [ "SPEED", "HEADING", "MODE", "turn", "acceleration",
       'current', 'rain', 'snowfall', 'wind_force', 'wind_direc',
        "resist_ratio","change_x_factor", "change_y_factor"]# 
static_categorical_feature = ["is_weekday", 'direction',"season", "departure_hour"] # ScheduleType #"adversarial"
y_cols = ["FC","SOG","LONGITUDE","LATITUDE"]
#action: speed, heading, mode, turn

config = InformerConfig.from_pretrained("huggingface/informer-tourism-monthly", prediction_length=prediction_horizon,
        context_length=context_length, input_size=len(y_cols), num_time_features=len(time_feature),
        num_dynamic_real_features = len(dynamic_real_feature), num_static_real_features = len(static_categorical_feature),
        lags_sequence=[1], num_static_categorical_features=0, feature_size=30)
model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly",
                                                           config=config, ignore_mismatched_sizes=True).to(device)

class GRU_update(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=4, num_layers=1, prediction_horizon=5):
        super().__init__()
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
        xy = torch.zeros(size=(past_time_features.shape[0], 1, self.output_size)).float().to(device)
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


trainset = vessel_data(train = True, train_test_split = 0.8, rand_seed=2)
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, drop_last=True)

validset = vessel_data(train = False, train_test_split = 0.8, rand_seed=2)
validloader = DataLoader(validset, batch_size = batch_size, drop_last=True)

testset = vessel_data(train = False, test=True, train_test_split = 0.8, rand_seed=2)
testloader = DataLoader(testset, batch_size = batch_size, drop_last=True)
testTripId = []

def get_testTripId():
    return testset.get_testTripId()

testTripId = get_testTripId()

model = model.float()

# gru = GRU_update(4, hidden_size=300, output_size = 4, num_layers=1, prediction_horizon=5).to(device)
gru = GRU_update(4, hidden_size=350, output_size = 4, num_layers=1, prediction_horizon=5).to(device)

#  train the model
def train_model(trainloader, testloader, validloader, model, model_name = "transformer"):
    mses = []
    r2s = []
    losses = []
    # valid_mses = []
    # valid_r2s = []
    actuals = []
    predictions = []
    train_mses = []
    train_r2s = []
    best_mse = 10000
    best_epoch = -1
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        train_mse = [[] for _ in range(4)]
        train_r2 = [[] for _ in range(4)]
        gru.train()
        model.train()
        for _, data in enumerate(trainloader):
            act = []
            pred = []
            values, time_features, static_real_features, future_features, actions  = data
            for t in range(sequence_length, values.shape[1]-prediction_horizon):
                # for i in range(prediction_horizon):
                i = (t-sequence_length)%prediction_horizon
                if i==0:
                    values_tmp = torch.clone(values[:, t-sequence_length:t+prediction_horizon*2])
                    time_features_tmp = torch.clone(time_features[:, t-sequence_length:t+prediction_horizon*2])
                    # print(values_tmp.shape, time_features_tmp.shape)
                else:
                    predicted = predicted.detach()
                    values_tmp[:, sequence_length-1 + i, :] =  predicted[:, 0, :]
                    acceleration = predicted[:, 0, 1]- values_tmp[:, sequence_length-2 + i, 1]
                    time_features_tmp[:, sequence_length-1 + i, 5] = acceleration
                    
                future_time_features = time_features_tmp[:, sequence_length+i-prediction_horizon: sequence_length+i]
                future_time_features[:, 0, 1:5] = time_features_tmp[:, sequence_length + i, 1:5]  # from speed to turn
                future_time_features[:, :, 0] = future_time_features[:, :, 0]+prediction_horizon/120

                future_values = values_tmp[:, sequence_length+i : sequence_length+i+prediction_horizon]
                past_values = values_tmp[:, i: sequence_length+i]
                past_time_features = time_features_tmp[:, i: sequence_length+i]
                
                # train
                past_observed_mask = torch.ones(past_values.shape).to(device)
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                tf_out = model(past_values=past_values, past_time_features=past_time_features, static_real_features=static_real_features,
                            past_observed_mask=past_observed_mask, future_values=future_values, future_time_features=future_time_features, output_hidden_states=True)
                loss1 = tf_out.loss
                # loss1.backward()
                
                with torch.no_grad():
                    predicted_tf = model.generate(past_values=past_values, past_time_features=past_time_features, static_real_features=static_real_features,
                                  past_observed_mask=past_observed_mask, future_time_features=future_time_features).sequences.mean(dim=1)
                predicted = gru(predicted_values = predicted_tf, past_time_features=past_time_features)


                loss2 = criterion(predicted, future_values).log()
                
                total_loss = 0.3*loss1 + 0.7*loss2
                total_loss.backward()

                optimizer1.step()
                optimizer2.step()
                print("loss1: ", loss1.item(), "loss2: ", loss2.item(), "total_loss: ", total_loss.item())

                # training loss
                actual = values[:, t, :].detach().cpu().numpy()
                yhat = predicted[:, 0, :].detach().cpu().numpy()
                act.append(actual)
                pred.append(yhat)
                # loss = prediction.loss
                epoch_loss += loss2.item()
            
            # training mse
            act = np.stack(act, axis=1)
            pred = np.stack(pred, axis=1)
            for i in range(4):
                t_mse = mean_squared_error(act[:,i], pred[:, i])
                # print(train_mse)
                train_mse[i].append(t_mse)
                actual_l = act[:, i].swapaxes(1, 0).reshape([-1, 1])
                predicted_l = pred[:, i].swapaxes(1, 0).reshape([-1, 1])
                t_r2 = r2_score(actual_l, predicted_l)
                train_r2[i].append(t_r2)
            actuals.append(act)
            predictions.append(pred)
       
        train_mse = [ sum(x)/len(x) for x in train_mse]
        train_r2 = [ sum(x)/len(x) for x in train_r2] 

        #step the scheduler with validation loss
        mse, r2 = evaluate_model(testloader, model)
        scheduler1.step(sum(mse)/4)
        scheduler2.step(sum(mse)/4)
        
        
        # best epoch
        if (sum(mse)/4 < best_mse):
            best_mse = sum(mse)/4
            best_epoch = epoch
       
        epoch_loss = epoch_loss / len(trainloader) / (data_len-25)
        print('Epoch %d / %d --- Loss: %.8f' % (epoch, n_epochs, epoch_loss))
        # mse, r2 = evaluate_model(testloader, model)
        print('train MSES: {}'.format(train_mse))
        print('train R2s:  {}'.format(train_r2))
        
        print('valid MSES: {}'.format(mse))
        print('valid R2s:  {}'.format(r2))

        mses.append(mse)
        r2s.append(r2)
        losses.append(epoch_loss)
        train_mses.append(train_mse)
        train_r2s.append(train_r2)
        wandb.log(
            {"Epoch": epoch, "Train_Loss": train_mse, "Train_R2": train_r2, "Valid_Loss": mse, "Valid_R2": r2}
        )
        # if (epoch == best_epoch-1):
        #     print("valid mses: ", mses)
        #     print("valid r2s: ", r2s)
        #     print("losses: ", losses)
        #     print("train_mses: ", train_mses)
        #     print("train_r2s: ", train_r2s)
        print("best epoch: ", best_epoch)
        path = "data/Checkpoints/{}/{}_checkpoint{}.pt".format(model_name, model_name, epoch)
        torch.save(model.state_dict(), path)
        path = "data/Checkpoints/{}/{}_checkpoint{}_gru.pt".format(model_name,model_name, epoch)
        torch.save(gru.state_dict(), path)
    
    path = "data/Checkpoints/Best/best_{}_checkpoint{}.pt".format( model_name, best_epoch)
    torch.save(model.state_dict(), path)
    path = "data/Checkpoints/Best/best_{}_checkpoint{}_gru.pt".format( model_name, best_epoch)
    torch.save(model.state_dict(), path)
    return mses, r2s, losses, train_mses, train_r2s, best_epoch



def evaluate_model(testloader, model, mode="valid", i=0):
    gru.eval()
    model.eval()
    predictions = []
    actuals = []
  
    mses = [[] for _ in range(4)]
    r2s =[[] for _ in range(4)]
    with torch.no_grad():
        for _, data in enumerate(testloader):
            outputs = None
            act = []
            pred = []
            values, time_features, static_real_features, future_features, actions  = data
            
            values_tmp = torch.clone(values)
            for t in range(sequence_length, values.shape[1]-prediction_horizon):
                future_values = values[:, t : t+prediction_horizon]
                if outputs != None:
                    predicted = outputs.detach()
                    values_tmp[:, t-1, :] =  predicted[:, 0, :]
                    acceleration = predicted[:, 0, 1]- values_tmp[:, t-2, 1]
                    time_features[:, t-1, 5] = acceleration


                future_time_features = time_features[:, t-prediction_horizon: t]
                future_time_features[:, 0, 1:5] = time_features[:, t, 1:5]  # from speed to turn
                future_time_features[:, :, 0] = future_time_features[:, :, 0]+prediction_horizon/120

                future_values = values[:, t : t+prediction_horizon]
                past_values = values_tmp[:, t-sequence_length: t]
                past_time_features = time_features[:, t-sequence_length: t]


                past_observed_mask = torch.ones(past_values.shape).to(device)
                predicted_tf = model.generate(past_values=past_values, past_time_features=past_time_features, static_real_features=static_real_features,
                                        past_observed_mask=past_observed_mask, future_time_features=future_time_features).sequences.mean(dim=1)
                outputs = gru(predicted_values = predicted_tf, past_time_features=past_time_features)
                

                yhat = outputs[:, 0, :].detach().cpu().numpy()
                actual = values[:, t, :].detach().cpu().numpy()

                act.append(actual)
                pred.append(yhat)

                

            act = np.stack(act, axis=1)
            pred = np.stack(pred, axis=1)
            for i in range(4):
                mse = mean_squared_error(act[:,i], pred[:, i])
                mses[i].append(mse)
                actual_1 = act[:, i].swapaxes(1, 0).reshape([-1, 1])
                yhat_1 = pred[:, i].swapaxes(1, 0).reshape([-1, 1])
                r2 = r2_score(actual_1, yhat_1)
                r2s[i].append(r2)
                
            
            predictions.append(pred)
            actuals.append(act)

    
    mses = [ sum(x)/len(x) for x in mses]
    r2s = [ sum(x)/len(x) for x in r2s]

    if mode=="test":
      return mses, r2s, actuals, predictions
    # calculate mse
    return mses, r2s
    #actuals[:20], predictions[:20]



iter = 15
iter = iter+1
criterion = torch.nn.MSELoss()

print(iter)
n_epochs = 80
best_epoch = 0
optimizer1 = torch.optim.Adam(model.parameters(), lr= 1e-4)
optimizer2 = torch.optim.Adam(gru.parameters(), lr= 1e-4)

# scheduler1 = StepLR(optimizer1, step_size=5, gamma=0.5)
# scheduler2 = StepLR(optimizer2, step_size=5, gamma=0.5)
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min', factor=0.5, patience=2, verbose=True, threshold=1e-4)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', factor=0.5, patience=2, verbose=True, threshold=1e-4)

# train
model_name = "Model_Iter_{}".format(iter)
if not os.path.exists("data/Checkpoints/{}".format(model_name)):
    os.makedirs("data/Checkpoints/{}".format(model_name))
   
if not os.path.exists("data/Checkpoints/Best"):
     os.makedirs("data/Checkpoints/Best")

wandb.login()
wandb.init(project="trip_loss", name="model_iter_{}".format(iter))


mses, r2s, losses, train_mses, train_r2s, best_epoch = train_model(trainloader, validloader, testloader, model, model_name)
# best_epoch = 54

# test

model.load_state_dict(torch.load("data/Checkpoints/{}/{}_checkpoint{}.pt".format(model_name,model_name, best_epoch), map_location=torch.device("cpu")))
gru.load_state_dict(torch.load("data/Checkpoints/{}/{}_checkpoint{}_gru.pt".format(model_name,model_name, best_epoch), map_location=torch.device("cpu")))

test_mses, test_r2s, test_actuals, test_predictions = evaluate_model(testloader, model, mode="test")


print("Best epoch at: ", best_epoch, "mse: " ,test_mses, "r2s: ", test_r2s)


def print_plot(best_epoch = 0, t_mses=[], t_r2s=[], v_mses=[],v_r2s=[],t_osses=[]):

    train_mses = np.array(t_mses)
    train_r2s = np.array(t_r2s)
    mses = np.array(v_mses)
    r2s = np.array(v_r2s)
    losses = np.array(t_osses)
    # valid_mses = np.array(valid_mses)

    # print("------ Print Info ------")
    # print("train_mses: ", train_mses)
    # print("train_r2s: ", train_r2s)
    # print("validation mses: ", mses)
    # print("validation r2s: ", r2s) 
    # print("epoch losses:", losses)
    # print("best epoch: ", best_epoch, "test mses: ", test_mses, "test r2s: ", test_r2s)

    # # load state dict
    load_iter = iter
    load_cp = best_epoch


    # print("Load Checkpoint: gru_{}_checkpoint{}".format(load_iter, load_cp))
 


        
    fig = plt.figure()
    grid = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)
    plt.title(f'iter_{load_iter} ')

    x = np.arange(1, n_epochs, 1)

    ax0 = plt.subplot(grid[0, 0])
    ax1 = plt.subplot(grid[0, 1:])
    ax2 = plt.subplot(grid[1, :1])
    ax3 = plt.subplot(grid[1, 1:])

    ax0.plot(x,mses[1:,0], label='valid', color='blue')
    # ax0.plot(x,valid_mses[1:,0], label='test', color='green')
    ax0.plot(x,train_mses[1:,0], label='train', color='black')

    ax1.plot(x,mses[1:,1], label='valid', color='blue')
    # ax1.plot(x,valid_mses[1:,1], label='test', color='green')
    ax1.plot(x,train_mses[1:,1], label='train', color='black')

    ax2.plot(x,mses[1:,2], label='valid', color='blue')
    # ax2.plot(x,valid_mses[1:,2], label='test', color='green')
    ax2.plot(x,train_mses[1:,2], label='train', color='black')

    ax3.plot(x,mses[1:,3], label='valid', color='blue')
    # ax3.plot(x,valid_mses[1:,3], label='test', color='green')
    ax3.plot(x,train_mses[1:,3], label='train', color='black')

    ax0.set_title('FC')
    ax1.set_title('SOG')
    ax2.set_title('LONGITUDE')
    ax3.set_title('LATITUDE')

    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()


    plot_filename = f'Losses_Plot_iter_{load_iter} , .png'

    plt.savefig("Plot/{}".format(plot_filename))
    wandb.log({f'Losses_Plot_iter_{load_iter}': wandb.Image("Plot/{}".format(plot_filename))})


# print_plot(best_epoch, train_mses, train_r2s, mses, r2s, losses)



