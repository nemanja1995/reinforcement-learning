import random
import numpy as np
import gym
import imageio  # write env render to mp4
import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import os
'''
Original paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- DQN model with Dense layers only
- Model input is changed to take current and n previous states where n = time_steps
- Multiple states are concatenated before given to the model
- Uses target model for more stable training
- More states was shown to have better performance for CartPole env
'''

LOGS_PATH = "data/logs"
MODELS_PATH = "data/models"
TEST_MODELS_PATH = "data/test_models"


class DQN:
    def __init__(
            self,
            env,
            memory_cap=1000,
            time_steps=3,
            gamma=0.85,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            learning_rate=0.005,
            batch_size=32,
            tau=0.125
    ):
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_shape = env.observation_space.shape
        self.time_steps = time_steps
        self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))

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

        self.summaries = {}

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_shape[0] * self.time_steps, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def update_states(self, new_state):
        # move the oldest state to the end of array and replace with new state
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        self.stored_states[-1] = new_state

    def act(self, test=False):
        states = self.stored_states.reshape((1, self.state_shape[0] * self.time_steps))
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        epsilon = 0.01 if test else self.epsilon  # use epsilon = 0.01 when testing
        q_values = self.model.predict(states)[0]
        self.summaries['q_val'] = max(q_values)
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # sample random action
        return np.argmax(q_values)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        states, action, reward, new_states, done = map(np.asarray, zip(*samples))
        batch_states = np.array(states).reshape(self.batch_size, -1)
        batch_new_states = np.array(new_states).reshape(self.batch_size, -1)
        batch_target = self.target_model.predict(batch_states)
        q_future = self.target_model.predict(batch_new_states).max(axis=1)
        batch_target[range(self.batch_size), action] = reward + (1 - done) * q_future * self.gamma
        hist = self.model.fit(batch_states, batch_target, epochs=1, verbose=0)
        self.summaries['loss'] = np.mean(hist.history['loss'])

    def target_update(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        # save model to file, give file name with .h5 extension
        self.model.save(fn)

    def load_model(self, fn):
        # load model from .h5 file
        self.model = tf.keras.models.load_model(fn)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def train(self, max_episodes=10, max_steps=500, save_freq=10):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = LOGS_PATH + '/DQN_basic_time_step{}/'.format(self.time_steps) + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, episode, steps, epoch, total_reward = True, 0, 0, 0, 0
        while episode < max_episodes:
            if steps >= max_steps:
                print("episode {}, reached max steps".format(episode))
                self.save_model(MODELS_PATH + "/dqn_basic_maxed_episode{}_time_step{}.h5".format(episode, self.time_steps))

            if done:
                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)

                self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))
                print("episode {}: {} reward".format(episode, total_reward))

                if episode % save_freq == 0:  # save model every n episodes
                    self.save_model(MODELS_PATH + "/dqn_basic_episode{}_time_step{}.h5".format(episode, self.time_steps))

                done, cur_state, steps, total_reward = False, self.env.reset(), 0, 0
                self.update_states(cur_state)  # update stored states
                episode += 1

            action = self.act()  # model determine action, states taken from self.stored_states
            new_state, reward, done, _ = self.env.step(action)  # perform action on env
            # modified_reward = 1 - abs(new_state[2] / (np.pi / 2))  # modified for CartPole env, reward based on angle
            prev_stored_states = self.stored_states
            self.update_states(new_state)  # update stored states
            self.remember(prev_stored_states, action, reward, self.stored_states, done)  # add to memory
            self.replay()  # iterates default (prediction) model through memory replay
            self.target_update()  # iterates target model

            total_reward += reward
            steps += 1
            epoch += 1

            # Tensorboard update
            with summary_writer.as_default():
                if len(self.memory) > self.batch_size:
                    tf.summary.scalar('Stats/loss', self.summaries['loss'], step=epoch)
                tf.summary.scalar('Stats/q_val', self.summaries['q_val'], step=epoch)
                tf.summary.scalar('Main/step_reward', reward, step=epoch)

            summary_writer.flush()

        self.save_model(MODELS_PATH + "/dqn_basic_final_episode{}_time_step{}.h5".format(episode, self.time_steps))

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            action = self.act(test=True)
            new_state, reward, done, _ = self.env.step(action)
            self.update_states(new_state)
            rewards += reward
            if render:
                video.append_data(self.env.render(mode='rgb_array'))
        video.close()
        return rewards


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 500
    dqn_agent = DQN(env, time_steps=5)
    # dqn_agent.load_model("basic_models/time_step4/dqn_basic_episode50_time_step4.h5")
    # dqn_agent.load_model(TEST_MODELS_PATH + "/dqn_basic_episode50_time_step4.h5")
    # dqn_agent.load_model(MODELS_PATH + "/dqn_basic_episode20_time_step5.h5")
    # rewards = dqn_agent.test()
    # print("Total rewards: ", rewards)
    dqn_agent.train(max_episodes=200)
