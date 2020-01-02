import numpy as np
from collections import deque


class Replay:
    def __init__(self, length):
        self.experience = {
            'state': deque(maxlen=length),
            'action': deque(maxlen=length),
            'reward': deque(maxlen=length),
            'next_state': deque(maxlen=length),
            'done': deque(maxlen=length)
        }

    @property
    def size(self):
        return len(self.experience['state'])

    def append(self, state, action, reward, next_state, done):
        self.experience['state'].append(state)
        self.experience['action'].append(action)
        self.experience['reward'].append(reward)
        self.experience['next_state'].append(next_state)
        self.experience['done'].append(done)

    def sample(self, size):
        indexes = np.random.randint(0, len(self.experience['state']), size=size)
        actions = np.asarray([self.experience['action'][i] for i in indexes])
        rewards = np.asarray([self.experience['reward'][i] for i in indexes])
        states = [self.experience['state'][i] for i in indexes]
        next_states = [self.experience['next_state'][i] for i in indexes]
        done = np.asarray([self.experience['done'][i] for i in indexes])
        for index, s in enumerate(states):
            states[index] = np.asarray(states[index])
            next_states[index] = np.asarray(next_states[index])
        states = np.asarray(states)
        next_states = np.asarray(next_states)
        return states, actions, rewards, next_states, done
