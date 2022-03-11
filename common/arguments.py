import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--n-agents", type=str, default=2, help="numbers of the agents")
    parser.add_argument("--opp-agents", type=str, default=1, help="numbers of the agents")
    parser.add_argument("--opp-sample-num", type=str, default=10, help="numbers of the agents")

    parser.add_argument("--state-dim", type=str, default=28, help="numbers of the state dim")
    parser.add_argument("--action-dim", type=str, default=5, help="numbers of the action dim")
    parser.add_argument("--num-quant", type=str, default=5, help="numbers of the quant")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=200000, help="number of time steps")

    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-3, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-4, help="learning rate of critic")
    parser.add_argument("--lr-entropy", type=float, default=3e-4, help="learning rate of entropy")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=3, help="number of episodes to optimize at the same time")
    #parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=200000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--target_update_interval", type=int, default=20,help="update target network once every time this many episodes are completed")

    parser.add_argument("--update-interval", type=int, default=10,
                        help="update network once every time this many episodes are completed")
    parser.add_argument("--start_steps", type=int, default=2000,help="start time step")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    # episolon decay
    parser.add_argument("--epsilon_ini", type=float, default=0.5, help="initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="minimum epsilon")
    parser.add_argument("--epsilon_decay_episode", type=int, default=500000,
                        help="number of episodes for decaying epsilon")
    parser.add_argument("--load_existing", type=bool, default=False, help="load existing network if true")
    parser.add_argument("--load_network_index", type=int, default=1, help="index of network to be loaded")
    parser.add_argument("--clip", type=float, default=5, help="clip value")
    args = parser.parse_args()


    return args
