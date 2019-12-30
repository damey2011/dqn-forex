import random
import tensorflow as tf
import numpy as np
import gym
from collections import deque
from tensorflow import keras

layers = keras.layers
optimizers = keras.optimizers
models = keras.models


class DeepQAgent:
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        self.action_space = self.env.action_space.n
        self.state_space = 3  # not really always 3, we are just splitting that of taxi to 3 since they are whole nos
        self.episodes = 200

        # Hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 0.1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 100

        # Experience replay
        self.experience = deque(maxlen=2000)

        # NN models
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.model.load_weights('./save_model/deepqtaxi_weights.h5')
        self.update_target_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, activation='relu', input_dim=self.state_space))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_space, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def update_model(self):
        if self.batch_size > len(self.experience):
            return
        mini_batch = np.array(random.sample(self.experience, self.batch_size))

        # states = np.hstack(mini_batch[:, 0]).reshape(self.batch_size, self.state_space)
        # # actions = mini_batch[:, 1]
        # # dones = mini_batch[:, 4]
        # # rewards = mini_batch[:, 2]
        # next_states = np.hstack(mini_batch[:, 3]).reshape(self.batch_size, self.state_space)
        #
        # target_output = self.model.predict(states)
        #
        # # Todo: Optimize this loop by doing bulk prediction and put everything into lists
        #
        # for index, replay in enumerate(mini_batch):
        #     try:
        #         next_is_done = mini_batch[index + 1][4]
        #     except IndexError:
        #         next_is_done = mini_batch[index][4]
        #
        #     if next_is_done:
        #         target_output[index][replay[1]] = replay[2]
        #     else:
        #         target_value = self.target_model.predict(next_states[index])[0]
        #         target_output[index][replay[1]] = replay[2] + (self.discount_factor * np.amax(target_value))
        #
        # self.model.fit(states, target_output, batch_size=self.batch_size, epochs=1, verbose=1)

        update_input = np.zeros((self.batch_size, self.state_space))
        update_target = np.zeros((self.batch_size, self.state_space))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def normalize_state(self, state):
        # This was added for the sake of deepqtaxi
        state = str(state)
        if len(state) < self.state_space:
            state = [int(x) for x in list(('0' * (self.state_space - len(state))) + state)]
        else:
            state = [int(x) for x in list(str(state))]
        return state

    def predict(self, state):
        nn_output = self.model.predict(state)
        return np.argmax(nn_output[0])

    def play(self):
        total_penalties = 0

        for i in range(self.episodes):
            state = self.env.reset()
            done = False

            score = 0
            steps_taken = 0
            penalties = 0

            while not done:
                if self.epsilon < random.random():
                    state_ = self.normalize_state(state)
                    action = self.predict([state_])
                else:
                    action = random.randrange(self.action_space)

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                next_state, reward, done, info = self.env.step(action)
                self.experience.append([self.normalize_state(state), action, reward, self.normalize_state(next_state), done])
                state = next_state
                score += reward
                steps_taken += 1
                if reward == -10:
                    penalties += 1
                self.update_model()

            if done:
                self.update_target_model()

            print(f'Played Episode {i} with {penalties} penalties and {steps_taken} moves.')

            total_penalties += penalties

            if i % 5 == 0:
                self.model.save_weights("./save_model/deepqtaxi_weights.h5")


DeepQAgent().play()
