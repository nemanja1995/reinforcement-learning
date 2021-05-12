from src.dqnCartPool.controllers.CartController import CartController
from src.dqnCartPool.CartPool import CartPole

import numpy as np


class InitCartController(CartController):
    def __init__(self):
        super().__init__()

    def act(self, state):
        action = np.random.randint(0, self.cart_pole.action_space_size)
        return action

