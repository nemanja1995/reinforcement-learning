from src.dqnCartPool.controllers.CartController import CartController
from src.dqnCartPool.controllers.InitCartController import InitCartController
from src.dqnCartPool.controllers.DQNController import DQNController


if __name__ == "__main__":
    cart_pole = DQNController()
    cart_pole.play(1)
