import threading

from src.dqnCartPoolM.controllers.CartController import CartController
from src.dqnCartPoolM.CartPool import CartPole
import numpy as np
from pynput.keyboard import Key, Listener


class KeyboardThread(threading.Thread):
    def __init__(self, name='keyboard-input-thread'):
        # self.input_cbk = input_cbk
        self.action = 0
        super(KeyboardThread, self).__init__(name=name)
        self.start()

    def run(self):
        with Listener(on_press=self.on_press) as listener:
            listener.join()
            # waits to get input + Return

    def on_press(self, key):
        # print('{0} pressed'.format(key))
        if key == Key.right:
            self.action = 1
        elif key == Key.left:
            self.action = 0


class PlayerCartController(CartController):
    def __init__(self, label="", log_path="data/logs",):
        super().__init__(label=label, log_path=log_path)
        self.action = 0
        # Collect events until released
        self.kthread = KeyboardThread()

    def act(self):
        return self.kthread.action

    def episode_start(self):
        self.cart_pole.render = True

    def play(self, num_episodes):
        print("Use arrows to control cart")
        super().play(num_episodes=num_episodes)

