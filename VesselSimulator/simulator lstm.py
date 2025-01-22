import gym
import numpy as np
import pickle
import torch
import torch.nn as nn


class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(AutoregressiveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

class CustomEnv(gym.Env):
    def __init__(self, model_path, start_trip_path, scaler_path = None, max_allowed=130, threshold=0.01, onetime_reward=5, device = None):
        super(CustomEnv, self).__init__()
        self.input_size = 22
        self.output_size = 18
        hidden_size = 128
        num_layers = 2

        self.model = AutoregressiveLSTM(self.input_size, self.output_size, hidden_size, num_layers).to(device)
        
        # Load model, scaler, and start trip data
        with open(model_path, 'rb') as f:
            self.model.load_state_dict(torch.load(f))
        with open(start_trip_path, 'rb') as f:
            self.start_trip = pickle.load(f)['X1']
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)['scaler']
        else:
            self.scaler = None
        
        # Hyperparameters
        self.max_allowed = max_allowed # max time allowed for trip
        self.threshold = threshold # threshold for close enough to destination
        self.onetime_reward = onetime_reward # reward for reaching the destination ontime
        
        # Action space: SPEED, HEADING (continuous), MODE (discrete)
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.action_space_mode = gym.spaces.Discrete(2)
        
        # Observation space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self.start_trip[0]),), dtype=np.float32)
        
        # Environment state
        self.state = None
        self.elapsed_time = 0
        self.hidden = None
        self.device = device
    
    def step(self, action):
        info = {}
        # Split action into SPEED, HEADING, and MODE
        speed, heading = action[:2]
        mode = int(action[2])  # MODE is discrete
        
        # Update state with action
        self.state[:3] = torch.tensor([speed, heading, mode])
        inp = self.state.unsqueeze(0).unsqueeze(1).to(device)
        # Predict next state
        prediction, self.hidden = self.model(inp, self.hidden)

        if self.scaler:
            aug = torch.cat([self.state.unsqueeze(0), prediction.squeeze(0)], dim=1)
            org = self.scaler.inverse_transform(aug.cpu().detach().numpy())[0]
            info['state_org'] = org[:self.input_size]
            info['output_org'] = org[self.input_size:self.input_size+4]
        
        # Extract outputs
        SFC, SOG, LATITUDE, LONGITUDE = prediction[0][0][:4]
        
        # Compute rewards
        r1 = -SFC
        LATITUDE_g = self.state[-2]
        LONGITUDE_g = self.state[-1]
        if self.state[15] == 0:  # direction
            r2 = 1 / (1 + (LATITUDE - self.state[-4])**2 + (LONGITUDE - self.state[-3])**2)
        else:
            r2 = 1 / (1 + (LATITUDE - self.state[-2])**2 + (LONGITUDE - self.state[-1])**2)
        
        r3 = 0
        if (LATITUDE - LATITUDE_g)**2 + (LONGITUDE - LONGITUDE_g)**2 < self.threshold:
            r3 = (self.max_allowed - self.elapsed_time) * self.onetime_reward
        
        reward = r1 + r2 + r3
        
        # Increment time and check termination
        self.elapsed_time += 1
        done = self.elapsed_time > 1.5 * self.max_allowed or r3 > 0
        
        
        
        return self.state, reward, done, False, info
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        
        # Select a random trip_id and initialize state
        start_index = np.random.randint(len(self.start_trip))
        state = self.start_trip[start_index][:22] 
        self.state = torch.tensor(state, dtype=torch.float32).to(device)
        self.elapsed_time = 0
        return self.state
    
if __name__ == '__main__':
    model_path = './data/VesselSimulator/lstm_model.pth'
    scaler_path = './data/VesselSimulator/scaler.pkl'
    start_trip_path = './data/VesselSimulator/start_trip.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CustomEnv(model_path, start_trip_path, scaler_path = scaler_path, device = device)
    print(env.reset())
    print(env.step([0.5, 0.5, 0])) # SPEED, HEADING, MODE