import numpy as np
from typing import Dict
import torch
from torch import Tensor
import gym


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        data: Dict[str, np.ndarray],
        device: str = "cpu",
    ):
        n_transitions = data["state"].shape[0]

        self._d4rl_size = n_transitions
        self._myenv_size = buffer_size
        buffer_size += n_transitions
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._next_actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device
        self.load_d4rl_dataset(data)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["state"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["state"])
        self._actions[:n_transitions] = self._to_tensor(data["action"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_state"])
        self._next_actions[:n_transitions] = self._to_tensor(data["next_action"])
        self._rewards[:n_transitions] = self._to_tensor(data["reward"][..., None])
        self._dones[:n_transitions] = self._to_tensor(data["done"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        #print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int):
        indices = torch.randint(0, self._size, (batch_size,)).to(self._device)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._states[indices+1]
        next_actions = self._actions[indices+1]
        dones = self._dones[indices]

        #print all shapes
        #print("states", states.shape, "actions", actions.shape, "rewards", rewards.shape, "next_states", next_states.shape, "next_actions", next_actions.shape, "dones", dones.shape)

        return {
            "state": states,
            "action": actions,
            "next_state": next_states,
            "next_action": next_actions,
            "reward": rewards.squeeze(-1),
            "done": dones.squeeze(-1),
        }
    
    def get_one_d4rl(self):
        rnd_idx = np.random.randint(0, self._d4rl_size)
        dones = self._dones[rnd_idx]
        start = rnd_idx
        while start >= 0 and dones[start] == 0:
            start -= 1
        start += 1
        end = rnd_idx
        while end < self._d4rl_size and dones[end] == 0:
            end += 1
        return self._states[start], end - start


    def add_transition_np(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        next_action: np.ndarray,
        reward: float,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._next_actions[self._pointer] = self._to_tensor(next_action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._dones[self._pointer] = self._to_tensor(done)

        # | ---- _d4rl_size ---- | ---- _myenv_size ---- |

        self._pointer = (self._pointer + 1 - self._d4rl_size) % self._myenv_size + self._d4rl_size
        self._size = min(self._size + 1, self._buffer_size)

