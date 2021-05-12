from src.dqnCartPoolM.controllers.CartController import CartController
from src.dqnCartPoolM.CartPool import CartPole

import numpy as np


class InitCartController(CartController):
    def __init__(self):
        super().__init__()
        self.cart_pole.set_render_true()

    def act(self):
        action = np.random.randint(0, self.cart_pole.a_space_size)
        return action

