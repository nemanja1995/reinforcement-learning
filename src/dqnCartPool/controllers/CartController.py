from src.dqnCartPool.CartPool import CartPole


class CartController:
    def __init__(self):
        self.cart_pole = CartPole()

    def act(self, state):
        pass

    def update(self):
        pass

    def play(self, num_episodes):
        self.cart_pole.play_episodes(action_func=self.act, update_func=self.update, num=num_episodes)
