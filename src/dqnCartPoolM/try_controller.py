from src.dqnCartPoolM.controllers.CartController import CartController
from src.dqnCartPoolM.controllers.DqnCartController import DqnCartController
from src.dqnCartPoolM.controllers.InitCartController import InitCartController
from src.dqnCartPoolM.controllers.PlayerCartController import PlayerCartController


if __name__ == "__main__":
    # cart_pole = DqnCartController(train_model=True,
    #                               save_path="data/models/research_tactics",
    #                               print_model=True, label="ivica1",
    #                               epsilon_decay=0.995)

    # cart_pole = DqnCartController(train_model=False,
    #                               episode_record=True,
    #                               load_path="data/models/dqn_controller_model/model_max_episode_38_reward_500.0.h5",
    #                               label="bot")
    
    cart_pole = DqnCartController(train_model=False,
                                  episode_record=True,
                                  load_path="data/models/research_tactics/model_max_episode_33_reward_500.h5",
                                  label="bot",
                                  gamma=1)

    # cart_pole = DqnCartController(train_model=False,
    #                               episode_record=True,
    #                               load_path="data/models/research_tactics/model_max_episode_31_reward_500.h5",
    #                               # load_path="data/test_models/model_max_episode_38_reward_500.0.h5",
    #                               label="bot")
    
    # cart_pole = PlayerCartController(label="nemanja", log_path="data/bio_robots")
    # cart_pole = InitCartController()
    cart_pole.play(40)
