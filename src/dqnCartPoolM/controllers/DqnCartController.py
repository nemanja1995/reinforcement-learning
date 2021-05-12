import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


from src.dqnCartPoolM.CartPool import CartPole
from src.dqnCartPoolM.controllers.CartController import CartController


class DqnCartController(CartController):
    def __init__(self,
                 train_model=False,
                 gamma=0.85,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 learning_rate=0.005,
                 batch_size=32,
                 tau=0.125,
                 save_freq=10,
                 load_path=None,
                 save_path=None,
                 log_path="data/logs/dqn_v1",
                 episode_record=False,
                 print_model=False,
                 label="",
                 render_freq=10
                 ):
        super().__init__(episode_record=episode_record, label=label)
        # self.log_path = log_path
        self.render_freq = render_freq
        self.print_model = print_model
        self.save_path = save_path
        self.load_path = load_path
        self.save_freq = save_freq

        self.train_model = train_model

        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # amount of randomness in e-greedy policy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay  # exponential decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau  # target model update

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        if load_path:
            self.load_model(load_path=self.load_path)

        self.max_reward = 0

        # self.summaries = {}
        # self.data_log = {"reward": [], "loss": []}

        if not train_model:
            self.cart_pole.set_render_true()

    def act(self, test=False):
        states = self.cart_pole.stored_states.reshape((1, self.cart_pole.space_size * self.cart_pole.time_steps))
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        epsilon = 0.01 if test else self.epsilon  # use epsilon = 0.01 when testing
        q_values = self.model.predict(states)[0]
        self.summaries['q_val'] = max(q_values)
        if np.random.random() < epsilon:
            return self.cart_pole.env.action_space.sample()  # sample random action
        return np.argmax(q_values)

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.cart_pole.space_size * self.cart_pole.time_steps, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(self.cart_pole.a_space_size))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        if self.print_model:
            model.summary()
        return model

    def episode_start(self):
        super().episode_start()
        if self.train_model:
            self.cart_pole.set_render_false()
            if self.cart_pole.episode_num % self.render_freq == 0 and self.render_freq > 0:
                self.cart_pole.set_render_true()
        else:
            self.cart_pole.set_render_true()

    def update(self):
        if self.train_model:
            self.replay()  # iterates default (prediction) model through memory replay
            self.target_update()  # iterates target model

    def episode_end(self):
        super().episode_end()
        total_reward = self.cart_pole.total_reward
        episode_num = self.cart_pole.episode_num

        if self.save_path:
            # if episode_num % self.save_freq == 0:
            #     full_path = os.path.join(self.save_path, "model_episode_{}_reward_{}.h5".format(episode_num,
            #                                                                                     total_reward))
            #     self.save_model(full_path)

            if total_reward > self.max_reward or total_reward == self.max_reward:
                self.max_reward = total_reward
                full_path = os.path.join(self.save_path, "model_max_episode_{}_reward_{}.h5".format(episode_num,
                                                                                                    int(total_reward)))
                self.save_model(full_path)

    def replay(self):
        if len(self.cart_pole.memory) < self.batch_size:
            return

        samples = random.sample(self.cart_pole.memory, self.batch_size)
        states, action, reward, new_states, done = map(np.asarray, zip(*samples))
        batch_states = np.array(states).reshape(self.batch_size, -1)
        batch_new_states = np.array(new_states).reshape(self.batch_size, -1)
        batch_target = self.target_model.predict(batch_states)
        q_future = self.target_model.predict(batch_new_states).max(axis=1)
        batch_target[range(self.batch_size), action] = reward + (1 - done) * q_future * self.gamma
        hist = self.model.fit(batch_states, batch_target, epochs=1, verbose=0)
        self.summaries['loss'] = np.mean(hist.history['loss'])

    def end(self):
        if self.train_model:
            super().end()

    def target_update(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, save_path):
        if not self.train_model:
            return
        # save model to file, give file name with .h5 extension
        print("Model saved to: {}".format(save_path))
        self.model.save(save_path)

    def load_model(self, load_path):
        # load model from .h5 file
        self.model = tf.keras.models.load_model(load_path)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
