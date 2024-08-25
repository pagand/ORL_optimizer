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
        _buffer_size: int,
        data: Dict[str, np.ndarray],
        device: str,
    ):
        n_transitions = data["state"].shape[0]

        self._d4rl_size = n_transitions
        self._myenv_size = _buffer_size
        self._highreward_size = _buffer_size
        self._lowreward_size = _buffer_size
        buffer_size = n_transitions + 3 * _buffer_size
        self._buffer_size = buffer_size

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
        n_transitions = data["state"].shape[0]
        self._states[:n_transitions] = self._to_tensor(data["state"])
        self._actions[:n_transitions] = self._to_tensor(data["action"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_state"])
        self._next_actions[:n_transitions] = self._to_tensor(data["next_action"])
        self._rewards[:n_transitions] = self._to_tensor(data["reward"][..., None])
        self._dones[:n_transitions] = self._to_tensor(data["done"][..., None])
        self._myenv_start = n_transitions
        self._myenv_pointer = n_transitions
        self._highreward_start = n_transitions + self._myenv_size
        self._highreward_pointer = n_transitions + self._myenv_size
        self._lowreward_start = n_transitions + self._myenv_size + self._highreward_size
        self._lowreward_pointer = n_transitions + self._myenv_size + self._highreward_size

        #print(f"Dataset size: {n_transitions}")
        self.type_dic = {
                # (pointer, start, size, capacity)
                "d4rl": (0, 0, n_transitions, n_transitions),
                "myenv": (self._myenv_pointer, self._myenv_start, 0, self._myenv_size,), 
                "highreward": (self._highreward_pointer, self._highreward_start, 0, self._highreward_size),
                "lowreward": (self._lowreward_pointer, self._lowreward_start, 0, self._lowreward_size),
            }

    def sample(self, batch_size: int, ratio_d4rl: float, ratio_myenv: float, ratio_highreward: float, ratio_lowreward: float) -> Dict[str, Tensor]:
        type_size = {
            "d4rl": int(batch_size * ratio_d4rl),
            "myenv": int(batch_size * ratio_myenv),
            "highreward": int(batch_size * ratio_highreward),
            "lowreward": int(batch_size * ratio_lowreward),
        }
        indices = None
        for type in ("d4rl", "myenv", "highreward", "lowreward"):
            _pointer, _start, _size, _capacity = self.type_dic[type]
            #print("type", type, "pointer", _pointer, "start", _start, "size", _size, "capacity", _capacity)
            N = min(type_size[type], _size)
            if N == 0:
                continue
            _indices = torch.randint(_start, _start + _size - 1, (N,)).to(self._device)
            if indices is None:
                indices = _indices
            else:
                indices = torch.cat((indices, _indices))
        #print("indices", indices)

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
    
    def add_transition_np(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        next_action: np.ndarray,
        reward: float,
        done: bool,
        type: str,
    ):
        _pointer, _start, _size, _capacity = self.type_dic[type]
        
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[_pointer] = self._to_tensor(state)
        self._actions[_pointer] = self._to_tensor(action)
        self._next_states[_pointer] = self._to_tensor(next_state)
        self._next_actions[_pointer] = self._to_tensor(next_action)
        self._rewards[_pointer] = self._to_tensor(reward).unsqueeze(-1)
        self._dones[_pointer] = self._to_tensor(done).unsqueeze(-1)

        # | ---- _d4rl_size ---- | ---- _myenv_size ---- | --- highreward_size --- | --- lowreward_size --- |

        _pointer = (_pointer + 1 - _start) % _capacity + _start
        _size = min(_size + 1, _capacity)

        self.type_dic[type] = (_pointer, _start, _size, _capacity)

