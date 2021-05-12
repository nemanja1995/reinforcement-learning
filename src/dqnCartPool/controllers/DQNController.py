from collections import deque

from src.dqnCartPool.controllers.CartController import CartController
from src.dqnCartPool.CartPool import CartPole
from src.dqnCartPool.TimeStepInfo import TimeStepInfo

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np


class DQNController(CartController):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.005
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.model = self.create_model()
        self.train_data = None

    def act(self, state):
        action = self.predict()
        return action

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.cart_pole.space_size * self.cart_pole.num_time_step, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.cart_pole.action_space_size, activation="softmax"))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def update(self):
        self.batch_data()
        self.train()

    def train(self):
        if self.train_data.shape[0] < self.batch_size/3:
            return

        input_states = self.train_data[:, 0]
        # x = np.resize(x, new_shape=[self.train_data.shape[0], self.cart_pole.num_time_step * self.cart_pole.space_size])
        x = input_states[0]
        # for row in input_states[1:]:
        #     x = np.vstack([x, row])
        # reward = self.train_data[:, 1]
        # done = self.train_data[:, 2]
        x, reward, done, new_states, action = map(np.asarray, zip(*self.train_data))
        inv_done = 1 - done
        p_reward = self.model.predict(x)
        p_max_reward = np.max(p_reward, axis=-1)
        y = p_max_reward * inv_done * self.epsilon_decay + reward
        y = np.reshape(y, newshape=[1, y.shape])
        # x = x.astype(np.float32)
        # x = tf.convert_to_tensor(x)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=1)

    def get_states_batched(self):
        states = []
        replay_step = None
        for num, replay_step in enumerate(self.cart_pole.memory):
            replay_step: TimeStepInfo
            state = replay_step.state
            states.extend(state)

        reward = replay_step.reward
        done = replay_step.done
        new_state = replay_step.new_state
        action = replay_step.action

        states = np.array(states)
        states.resize([1, self.cart_pole.space_size * self.cart_pole.num_time_step], refcheck=False)
        new_states = np.roll(states, -1, axis=0)
        new_states[-1] = new_state
        if done:
            done = 1
        else:
            done = 0

        return states, reward, done, new_states, action

    def batch_data(self):
        state, reward, done, new_states, action = self.get_states_batched()
        new_data = np.array([state, reward, done, new_states, action])
        if self.train_data is None:
            self.train_data = new_data
        elif len(self.train_data) >= self.batch_size:
            self.train_data = np.roll(self.train_data, -1, axis=0)
            self.train_data[-1] = new_data
        else:
            self.train_data = np.vstack([self.train_data, new_data])

    def predict(self):
        x = self.get_states_batched()[0]
        action_distribution = self.model.predict(x)
        action = np.argmax(action_distribution)
        print("distribution: {} -> {}".format(action_distribution, action))
        return action
