import argparse
import random
import numpy as np
import torch
from pathlib import Path
from ConfigurationLoader import ConfigurationLoader


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

def train(input_file):

    """Train a agent using the parameters specified in the 
    input file.

    Parameters:
    - input_file = The file that contains the parameters to train the 
      agent.
    """

    agent = ConfigurationLoader(input_file).get_agent()
    agent.train()
    return agent

def test(input_file, agent = None):

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
    reward, ep_length= agent.test(1000)
    print(reward, ep_length)
    show_func = agent.environment.get_wrapper_attr('show_episode_result')
    show_func()

def main(args):

    input_file = args.filename[0]
    is_testing = args.test
    if (is_testing is None):
        agent = train(input_file)
    test(input_file)



if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main(parse_arguments())
