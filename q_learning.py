"""
Let's design a simulation of a self-driving cab. The major goal is to demonstrate, in a simplified environment, how you
can use RL techniques to develop an efficient and safe approach for tackling this problem.

The Smartcab's job is to pick up the passenger at one location and drop them off in another. Here are a few things that
we'd love our Smartcab to take care of:

Drop off the passenger to the right location.
Save passenger's time by taking minimum time possible to drop off
Take care of passenger's safety and traffic rules
"""
import os

import gym
import numpy as np


class QLearning:
    # Learnt from:
    # https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
    # Hyperparameters
    ALPHA = 0.1
    """α  (alpha) is the learning rate (0 < α ≤ 1) - Just like in supervised learning settings, α is the extent 
    to which our Q-values are being updated in every iteration."""
    GAMMA = 0.6
    """
    γ  (gamma) is the discount factor (0 ≤ γ ≤ 1) - determines how much importance we want to give to future rewards. 
    A high value for the discount factor (close to 1) captures the long-term effective award, whereas, a discount 
    factor of 0 makes our agent consider only immediate reward, hence making it greedy.
    """
    EPSILON = 0.1

    ENV = gym.make("Taxi-v3")

    ACTION_SPACE = ENV.action_space.n
    STATE_SPACE = ENV.observation_space.n

    Q_TABLE = np.zeros((STATE_SPACE, ACTION_SPACE))

    def __init__(self):
        os.environ.setdefault('TERM', 'xterm-color')

    def update_q_table(self, state, action, reward, next_state):
        old_q = self.Q_TABLE[state, action]
        # Get the maximum reward for the next state
        next_max = np.max(self.Q_TABLE[next_state])
        # Get the new q value by running it through the Q-Learning formula
        new_q = ((1 - self.ALPHA) * old_q) + (self.ALPHA * (reward + self.GAMMA * next_max))
        self.Q_TABLE[state, action] = new_q

    def train(self):
        print('Training the agent to play the taxi game')
        for i in range(5000):
            done = False
            state = self.ENV.reset()
            penalties = 0
            while not done:
                if np.random.randint(0, 1) < self.EPSILON:
                    # Explore
                    action = self.ENV.action_space.sample()
                else:
                    # Exploit
                    # Get the row of the state in the Q table, and check the action that has the max value
                    action = np.argmax(self.Q_TABLE[state])
                    # arg max is to get the index of the biggest action in that state row
                next_state, reward, done, info = self.ENV.step(action)
                # Update the Q-Table
                self.update_q_table(state, action, reward, next_state)
                if reward == -10:
                    penalties += 1

                state = next_state

            if i + 1 % 100 == 0:
                print('Episode {} reached'.format(i))

    def play(self):
        print('Now the agent will play 100 episodes from the knowledge gained')
        episodes = 100
        total_epoch, total_penalties = 0, 0
        for i in range(episodes):
            state = self.ENV.reset()
            epoch, penalties = 0, 0
            done = False
            while not done:
                action = np.argmax(self.Q_TABLE[state])
                state, reward, done, info = self.ENV.step(action)
                if reward == -10:
                    penalties += 1
                epoch += 1
            total_epoch += epoch
            total_penalties += penalties
            print(f'Made {epoch} moves to complete episode {i} with {penalties} penalties')

        # print('Results after {} episodes:')
        # print(f'Total Numbers of ')


QLearning().train()
QLearning().play()
