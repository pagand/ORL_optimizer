import gym
import numpy as np
import pickle
import torch
import torch.nn as nn
import time


class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0):
        super(AutoregressiveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout*0.1)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)  # Apply dropout
        out = self.fc(out)
        return out, hidden

class CustomEnv(gym.Env):
    def __init__(self, model_path, start_trip_path, scaler_path, max_allowed=130, goal_threshold=0.01, onetime_reward=5, device = None):
        super(CustomEnv, self).__init__()
        self.input_size = 22
        self.output_size = 18
        hidden_size = 128
        num_layers = 2

        self.model = AutoregressiveLSTM(self.input_size, self.output_size, hidden_size, num_layers).to(device)

        # print number of model parameters
        # print(f"\n\n\n\n\nNumber of model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        # Load model, scaler, and start trip data
        with open(model_path, 'rb') as f:
            self.model.load_state_dict(torch.load(f))
        with open(start_trip_path, 'rb') as f:
            content = pickle.load(f)
            self.start_trip = content['X1']
            self.state_avg = content['x_avg']

        with open(scaler_path, 'rb') as f:
            content = pickle.load(f)
            self.scaler = content['scaler']
            self.feature_columns = content['feature_columns']
            self.output_columns = content['output_columns']
        
        
        # Hyperparameters
        self.max_allowed = max_allowed # max time allowed for trip
        self.goal_threshold = goal_threshold # threshold for close enough to destination
        self.onetime_reward = onetime_reward # reward for reaching the destination ontime
        
        # Action space: SPEED, HEADING (continuous), MODE (discrete)
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.action_space_mode = gym.spaces.Discrete(2)
        
        # Observation space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self.start_trip[0]),), dtype=np.float32)
        
        # Environment state
        self.state = None
        self.device = device
        self.goal = [[49.19541486727186 ,-123.9551366316478 ], [49.3788352862298 , -123.27131442779599 ]] # for [direction 0, direction 1]
        self.max_distance = 0.708 # actual distance

    def reset(self, initial_state=None):
        seed = 10
        super().reset(seed=seed)
        np.random.seed(seed)
        
        # Select a random trip_id and initialize state
        if initial_state is not None:
            state = initial_state
        else:
            id = np.random.randint(len(self.start_trip))
            state = self.start_trip[id][:22] 
        self.state = torch.tensor(state, dtype=torch.float32)
        self.elapsed_time = 0
        self.hidden = None
        self.hidden_ar = None
        self.direction = int(state[9])
        self.state_avg_org = self.scaler.inverse_transform(self.state_avg)[self.direction]
        return self.state
    
    def step(self, action, next_state_measured=None): 
        '''
        IMPORTANT:
        - Saceled variables are:
            self.state, action, next_state_measured, prediction, info['prediction']
        - Original variables are:
            new_state, self.state_avg_org, info['state_org'], info['output_org']
        '''
        info = {}
        # check if all actions are between 0 and 1 otherwise raise error
        if not all(0<=a<=1 for a in action):
            raise ValueError('Action values should be between 0 and 1')
        # Split action into SPEED, HEADING, and MODE
        speed, heading = action[:2]
        mode = int(action[2])  # MODE is discrete
        self.state[:3] = torch.tensor([speed, heading, mode])

        # if we have a next_state_measured, we should use it to update the state
        if next_state_measured is not None:
            # use the state_measured for current state values
            self.state[3:self.input_size] = torch.tensor(next_state_measured, dtype=torch.float32)

            aug = np.concat((np.expand_dims(np.concat((action,next_state_measured) ),0), np.zeros((1, self.output_size))), axis=1)
            current_state = self.scaler.inverse_transform(aug)[0][:self.input_size]
            
            # update the env properties
            self.elapsed_time = current_state[11]
            # check if the direction is the same, raise error if not
            if self.direction != int(current_state[9]):
                raise ValueError('Direction changed during the trip')



        inp = self.state.unsqueeze(0).unsqueeze(1).to(self.device)

        # decide between AR and NAR
        if next_state_measured is None: # AR
            hidden = self.hidden_ar
        else:
            hidden = self.hidden

        # Predict next state
        prediction, hidden = self.model(inp, hidden)

        if next_state_measured is None: #AR
            self.hidden_ar = hidden
        else: # NAR
            # if NAR, both hidden and hidden_ar should be updated
            self.hidden = hidden
            self.hidden_ar = hidden

        
        info['prediction'] = prediction.squeeze(0).squeeze(0).cpu().detach().numpy()

        # compute the original values
        aug = torch.cat([self.state.unsqueeze(0), prediction.squeeze(0).cpu().detach()], dim=1).numpy()
        org = self.scaler.inverse_transform(aug)[0]

        
        # create a dictionary with self.feature_columns as keys and info['state_org'] as values
        info['state_org'] = dict(zip(self.feature_columns, org[:self.input_size]))
        info['output_org'] = dict(zip(self.output_columns, org[self.input_size:]))
        # update states
        '''
        '0 SPEED', '1 HEADING', '2 MODE', '3 ENGINE_FLOWTEMPA', '4 PITCH', '5 POWER', '6 STW',
       '7 WIND_ANGLE', '8 WIND_SPEED', '9 direction', '10 distance', '11 elapsed_time',
       '12 cumulative_SFC', '13 current', '14 pressure', '15 weathercode', '16 is_weekday',
       '17 effective_wind_factor', '18 PSFC', '19 PSOG', '20 PLAT', '21 PLON'
        '''


        self.elapsed_time += 1
        org[3:9] = self.state_avg_org[3:9]
        org[10] = np.sqrt((self.goal[self.direction][0]-info['output_org']['LAT'])**2+(self.goal[self.direction][1]-info['output_org']['LON'])**2)
        org[11] = self.elapsed_time
        org[12] += info['output_org']['SFC']
        org[13:18] = self.state_avg_org[13:18]
        org[18:22] = info['output_org']['SFC'], info['output_org']['SOG'], info['output_org']['LAT'], info['output_org']['LON']

        
        # convert to scaled values
        aug = self.scaler.transform(org.reshape(1, -1))[0]
        self.state = torch.tensor(aug[:self.input_size], dtype=torch.float32)
        reward = -info['prediction'][0]   # negative of SFC (scaled)
        reward+= max(0, self.max_distance-org[10])/self.max_distance   # reward for getting closer to the destination
        done = org[10]<self.goal_threshold
        reward+= 5 if done and org[11]<self.max_allowed else 0  # one time reward if reach the goal on time
        

        return self.state, reward, done, info
    
    
if __name__ == '__main__':
    model_path = './data/VesselSimulator/lstm_model_ar5.pth'
    scaler_path = './data/VesselSimulator/scaler.pkl'
    start_trip_path = './data/VesselSimulator/start_trip.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


    start = time.time()
    env = CustomEnv(model_path, start_trip_path, scaler_path = scaler_path, device = device)
    # t1 = time.time()
    # state = env.reset()
    # t2 = time.time()
    # state, reward, done, info = env.step([0.5, 0.5, 0]) # SPEED, HEADING, MODE
    # t3 = time.time()
    # print(t1-start, t2-t1, t3-t2)


    # test 2
    test_path = './data/VesselSimulator/data_test.pkl'
    with open(test_path, 'rb') as f:
        content = pickle.load(f)
        X1_test_state = content['X1'][0] 
        X1_test_target = content['X1'][1]

    i = 0 # select one test trip from 0 to 4 
    current_trip = X1_test_state[i, :, 1:] # ignoring the first state which is done state in dataset
    env.reset(current_trip[0][:22]) # without the done flag
    state, reward, done, info = env.step(current_trip[0][:3],current_trip[0][3:22]) # step 1
    state, reward, done, info = env.step(current_trip[1][:3],current_trip[1][3:22]) # step 2
    state, reward, done, info = env.step(current_trip[2][:3]) # step 2, AR for 3
    state, reward, done, info = env.step(current_trip[3][:3]) # step 2, AR for 4
    state, reward, done, info = env.step(current_trip[2][:3],current_trip[2][3:22]) # step 3

 

    


    
    






    