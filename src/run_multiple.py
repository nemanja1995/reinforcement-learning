from src.dqnCartPoolM.controllers.CartController import CartController
from src.dqnCartPoolM.controllers.DqnCartController import DqnCartController
from src.dqnCartPoolM.controllers.InitCartController import InitCartController
from src.dqnCartPoolM.controllers.PlayerCartController import PlayerCartController


if __name__ == "__main__":
    label = "gam2ma"
    num_tests = 10
    parameter = ""
    # test_value = [0.99925]
    # test_value = [0.90, 0.95, 0.97, 0.98, 0.99]
    test_value = [1.25, 1.5]
    for t_value in test_value:
        for i in range(num_tests):
            cart_pole = DqnCartController(train_model=True,
                                          # save_path="data/models/dqn_controller_model",
                                          save_path="data/models/research_tactics",
                                          print_model=True,
                                          label=label + "-" + str(t_value),
                                          # label=label,
                                          # epsilon_decay=t_value,
                                          gamma=t_value,
                                          render_freq=-1)

            cart_pole.play(40)
