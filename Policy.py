from abc import ABC, abstractmethod


class Policy(ABC):


    """
    Policy is an interface that specifies the methods that policy
    concrete classes must implement.
    """
    
    @abstractmethod
    def select_action(self, state, actions):
        pass
        
    @abstractmethod
    def update_exploration_rate(self, new_value: float):
        pass
