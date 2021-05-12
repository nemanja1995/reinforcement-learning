import os

import gym
import matplotlib.pyplot as plt
from datetime import datetime
import json

from src.dqnCartPoolM.CartPool import CartPole


class CartController:
    MAX_EPISODE_STEPS = 500

    def __init__(self, episode_record=False, log_path="data/logs", label=""):
        self.label = label
        self.log_path = log_path
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        env = gym.make("CartPole-v1")
        env._max_episode_steps = CartController.MAX_EPISODE_STEPS
        self.cart_pole = CartPole(env=env,
                                  update=self.update,
                                  act=self.act,
                                  episode_end=self.episode_end,
                                  episode_start=self.episode_start,
                                  log=False,
                                  render=False,
                                  record_episode=episode_record)
        self.summaries = {}
        self.data_log = {"reward": [], "loss": []}
        self.log_dir_path = ""

    def act(self):
        pass

    def update(self):
        pass

    def episode_end(self):
        total_reward = self.cart_pole.total_reward
        self.data_log["reward"].append(total_reward)
        loss = self.summaries.get("loss", 0)
        self.data_log["loss"].append(loss)

    def episode_start(self):
        pass

    def end(self):
        date = datetime.now()
        rewards = self.data_log.get("reward", [])
        loss = self.data_log.get("loss", [])
        plt.plot(rewards)
        plt.ylabel('Rewards')
        plt.xlabel('Episodes')
        if self.log_path:
            full_path = os.path.join(self.log_dir_path, "rewards_{}.png".format(date))
            plt.savefig(full_path)
        plt.show()

        plt.plot(loss)
        plt.ylabel('Loss')
        plt.xlabel('Episodes')
        if self.log_path:
            full_path = os.path.join(self.log_dir_path, "episodes_{}.png".format(date))
            plt.savefig(full_path)
        plt.show()

        if self.log_path:
            full_path = os.path.join(self.log_dir_path, "data_{}.json".format(date))
            fw = open(full_path, "w")
            json.dump({"rewards": rewards, "loss": loss}, fw)

    def play(self, num_episodes):
        log_dir = self.label + "_" + str(datetime.now()).replace(" ", "_")
        self.log_dir_path = os.path.join(self.log_path, log_dir)
        os.makedirs(self.log_dir_path)
        self.cart_pole.play_episodes(num_episodes=num_episodes)
        self.end()
