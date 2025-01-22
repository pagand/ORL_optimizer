import gym
import numpy as np
import pickle

class CustomEnv(gym.Env):
    def __init__(self, model_path, start_trip_path, scaler_path = None, max_allowed=130, threshold=0.01, onetime_reward=5):
        super(CustomEnv, self).__init__()
        
        # Load model, scaler, and start trip data
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(start_trip_path, 'rb') as f:
            self.start_trip = pickle.load(f)['X1']
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
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
    
    def step(self, action):
        info = {}
        # Split action into SPEED, HEADING, and MODE
        speed, heading = action[:2]
        mode = int(action[2])  # MODE is discrete
        
        # Update state with action
        self.state[:3] = [speed, heading, mode]
        
        # Predict next state
        prediction = self.model.predict([self.state])

        if self.scaler:
            state_org = self.scaler['scaler_X'].inverse_transform([self.state])
            output_org = self.scaler['scaler_Y'].inverse_transform(prediction)  # Inverse transform
            info['state_org'] = state_org
            info['output_org'] = output_org  # SFC, SOG, LATITUDE, LONGITUDE
        
        # Extract outputs
        SFC, SOG, LATITUDE, LONGITUDE = prediction[0]
        self.state[-4:] = [SFC, SOG, LATITUDE, LONGITUDE]
        
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
        self.state = np.array(self.start_trip[start_index])
        self.elapsed_time = 0
        return self.state
    
if __name__ == '__main__':
    model_path = './data/model_2019_2020.pkl'
    scaler_path = './data/scaler.pkl'
    start_trip_path = './data/start_trip.pkl'
    env = CustomEnv(model_path, start_trip_path, scaler_path = scaler_path)
    print(env.reset())
    print(env.step([0.5, 0.5, 0]))