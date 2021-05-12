from collections import deque

import gym
from gym.spaces import Box
import time

from src.dqnCartPool.TimeStepInfo import TimeStepInfo

REPLAY_LENGTH = 5


class CartPole:
    def __init__(self, num_time_step=5):
        self.env = gym.make("CartPole-v1")
        self.space = self.env.observation_space
        self.space: Box
        self.space_size = self.space.shape[0]
        self.num_time_step = num_time_step
        self.action_space_size = self.env.action_space.n

        self.memory = deque(maxlen=self.num_time_step)

    def update_memory(self, time_step: TimeStepInfo):
        if len(self.memory) > self.memory.maxlen:
            self.memory.popleft()
        self.memory.append(time_step)

    def play_episodes(self, action_func, update_func, num=1, render=True, log=True):
        for i in range(num):
            self.episode(action_func=action_func, update_func=update_func, render=render, log=log)

    @staticmethod
    def dummy_action_func(states):
        return 0

    @staticmethod
    def update_func():
        pass

    def episode(self, action_func=None, update_func=None, render=True, log=True):
        if not action_func:
            action_func = CartPole.dummy_action_func
        if not update_func:
            update_func = CartPole.update_func

        done = False
        state = self.env.reset()

        time_step = TimeStepInfo(state=state, reward=0, new_state=state, done=done, step_num=0, action=0)
        self.update_memory(time_step)
        total_reward = 0
        step_num = 0
        while not done:
            step_num += 1
            new_action = action_func(state)
            print("action: {}".format(new_action))
            new_state, reward, done = self.frame(new_action, render=render, log=log)
            total_reward += reward
            time_step = TimeStepInfo(state=state, reward=reward, new_state=new_state,
                                     done=done, step_num=step_num, action=new_action)
            self.update_memory(time_step)
            update_func()
            state = new_state

        self.env.close()
        print("-" * 50)
        print("total reward: {}".format(total_reward))
        print("-" * 50)

    def frame(self, action, render=True, log=True):
        new_state, reward, done, _ = self.env.step(action=action)
        new_state: Box
        if log:
            print("state: {}".format(new_state))
            print("reward: {} \t\t done: {}".format(reward, done))
            print("-" * 50)
        if render:
            self.env.render()
        time.sleep(0.1)
        return new_state, reward, done
