import argparse
import random
import numpy as np
import torch
import time
from pathlib import Path
from .ConfigurationLoader import ConfigurationLoader
from GLOMIP import GLOMIP


def parse_arguments ():
    args_parser = argparse.ArgumentParser(
        description= ("Create DRL Agent to learn how "
                      "to perform in a given enviroment."))
    args_parser.add_argument("filename",
                             metavar="F",
                             type=Path,
                             nargs=1,
                             help=("The path of the input file "
                                   "with the agent parameters."))
    args_parser.add_argument("-t",
                             "--test",
                             nargs="?",
                             help="Test an already trained model",
                             type=bool,
                             const=True)
    return args_parser.parse_args()

def train(input_file: Path):
    """Train an agent using the parameters specified in the input file.

    Args:
        input_file (Path): The path of the file that contains the
        parameters.

    Returns:
        DQLearning: The Trained Agent.
    """



    agent = ConfigurationLoader(input_file).get_agent()
    agent.train()
    return agent

def test_agent(input_file, agent = None):

    """Use a trained agent to test its performance. If the program is
    executed without the "--test" flag the model is initialized here.

    Parameters:
    - input_file = The path of the file containing the agent
      information.
    - agent = The initialized agent.
    """

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if (agent is None):
        agent = ConfigurationLoader(input_file).get_agent()
    agent.q_estimator.load_model()
    start_time = time.time()
    reward, ep_length, jumps = agent.test(100)
    print(f"Elapsed time = {time.time() - start_time }")
    print(reward, ep_length, jumps)
#    show_func = agent.environment.get_wrapper_attr('show_episode_result')
#    show_func()

def test_glomip(input_file):

    """Solve the environment using the ILP formulation.

    Parameters:
    - input_file = The path of the file containing the agent
      information.
    """

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    config = ConfigurationLoader(input_file).get_parameters()
    glomip = GLOMIP(config["env"])
    start_time = time.time()
    reward, ep_length, jumps = glomip.test(100)
    print(f"Elapsed time = {time.time() - start_time }")
    print(reward, ep_length, jumps)
#    show_func = glomip.env.get_wrapper_attr('show_episode_result')
#    show_func()

def main(args):

    input_file = args.filename[0]
    is_testing = args.test
    if (is_testing is None):
        agent = train(input_file)
    test_agent(input_file)
    # test_glomip(input_file)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main(parse_arguments())
