import random
import numpy as np
import imageio  # write env render to mp4
import datetime
from collections import deque
import tensorflow as tf
import gym
from gym.spaces import Box
import time

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


class CartPole:
    def __init__(
            self,
            env=None,
            memory_cap=1000,
            time_steps=3,
            act=None,
            update=None,
            episode_end=None,
            episode_start=None,
            render=True,
            log=True,
            record_episode=True,
            video_path="last_episode_replay.mp4"
    ):
        self.record_episode = record_episode
        if not env:
            env = gym.make("CartPole-v1")
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_shape = env.observation_space.shape
        self.time_steps = time_steps
        self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))

        self.space = self.env.observation_space
        self.space: Box
        self.space_size = self.space.shape[0]
        self.a_space_size = self.env.action_space.n
        self.video_recorder = None
        self.video_path = video_path

        if not act:
            act = self.dummy_act
        self.act = act

        if not update:
            update = self.dummy_update

        if not episode_end:
            episode_end = self.dummy_episode_end
        self.episode_end = episode_end

        if not episode_start:
            episode_start = self.dummy_episode_start
        self.episode_start = episode_start

        self.update = update
        self.render = render
        self.summaries = {}
        self.log = log

        self.episode_num = 0
        self.total_steps = 0
        self.total_reward = 0

    def set_render_true(self):
        self.render = True

    def set_render_false(self):
        self.render = False

    def update_states(self, new_state):
        # move the oldest state to the end of array and replace with new state
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        self.stored_states[-1] = new_state

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def dummy_act(self):
        return 0

    def dummy_update(self):
        pass

    def dummy_episode_end(self):
        pass

    def dummy_episode_start(self):
        pass

    def play_episodes(self, num_episodes=1, max_steps=500):
        self.episode_num = 0
        while self.episode_num < num_episodes:
            self.episode_start()
            steps = self.episode(max_steps=max_steps)
            print("Ep: {} -> reward: {}\t steps: {}".format(self.episode_num, self.total_reward, steps))
            self.episode_num += 1
            print("Epistode END")
            if self.render:
                for t in range(3):
                    print("Start in {}".format(3-t))
                    time.sleep(1)
            self.episode_end()

    def episode(self, max_steps=500):
        self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))
        done, cur_state, steps, self.total_reward = False, self.env.reset(), 0, 0
        self.update_states(cur_state)  # update stored states
        if self.record_episode:
            self.video_recorder = imageio.get_writer(self.video_path, fps=30)

        while not done:
            action = self.act()  # model determine action, states taken from self.stored_states
            new_state, reward, done, _ = self.env.step(action)  # perform action on env
            if self.render:
                if self.video_recorder:
                    self.video_recorder.append_data(self.env.render(mode='rgb_array'))
                else:
                    self.env.render()
            if self.log:
                print("[{}] state: {}".format(steps, new_state))
                print("reward: {} \t\t done: {}".format(reward, done))
                print("-" * 50)
            # modified_reward = 1 - abs(new_state[2] / (np.pi / 2))  # modified for CartPole env, reward based on angle
            prev_stored_states = self.stored_states
            self.update_states(new_state)  # update stored states
            self.remember(prev_stored_states, action, reward, self.stored_states, done)  # add to memory
            self.update()
            self.total_reward += reward
            steps += 1
            if self.render:
                time.sleep(.0)
            if max_steps < steps:
                done = True
        self.env.close()
        if self.video_recorder:
            self.video_recorder.close()
        return steps
