import torch

class Experience(object):


    """
    An Experience is a data structure that holds the following 
    information: (state, action, reward, new_state, priority).

    Attributes:
    - state: is the current state of the env.
    - action: the action taken in the current env.
    - reward: the reward associated to taking said action in the
      current environment.
    - next_state: the resulting state after applying the action.
    - done: a flag that states if a state is terminal.
    - priority: a numerical value used for prioritized memory 
      replay. Is a measurement of the importance of the experience.
    """
    
    def __init__(self, state, action, reward, next_state, done, priority: float = 9999):

        """
        Create a new Experience, i.e. a (s,a,r,s',d,p) tuple
        

        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.priority = priority

    def get_state(self) -> torch.Tensor:

        """
        Return a Tensor containing the state information.
        """

        state = torch.Tensor(self.state.values(), dtype = torch.float64)
        return state

    def __str__(self) -> str:

        """
        Parses the object information into a human-readable string
        """

        string = "Experience:\n"
        string += f"\tState: {self.state}\n" 
        string += f"\tAction: {self.action}\n" 
        string += f"\tReward: {self.reward}\n" 
        string += f"\tNext state: {self.next_state}\n" 
        string += f"\tIs terminal: {self.done}\n"
        string += f"\tPriority: {self.priority}\n"

        return string
