from collections import deque
import random
import torch
import numpy as np
from Experience import Experience


class ExperienceSampler:


    """
    This class is the responsible of providing the experience batches
    to the DQLearning agent.

    It samples the experiences from the experiences buffer according to
    a sampling policy.

    Experiences are tuples of (s,a,r,s',p) and the buffer is 
    implemented as a queue.
    """

    def __init__(self,
                 buffer_max_size,
                 sampling_policy: str = "simple",
                 device: str = "cpu",
                 epsilon: float = 1e-4,
                 alpha: float = 1) -> None:

        """

        """

        self.experience_buffer: deque[Experience] =\
            deque([], maxlen=buffer_max_size)
        self.device = device
        self.sampling_policy = sampling_policy
        if (self.sampling_policy == "per"):
            self.epsilon = torch.tensor(epsilon).to(self.device)
            self.alpha = torch.tensor(alpha).to(self.device)
            pass
        if (self.sampling_policy == "simple"):
            pass

    def add_experience(self, experience: Experience):

        """
        Store an experience into the buffer.
        """

        self.experience_buffer.append(experience)

    def sample_experience(self, n: int = 1) -> list:

        """
        Return a list of experiences sampled according to the 
        sampling policy.
        """

        if (self.sampling_policy == "simple"):
            sample = random.sample(self.experience_buffer, n)
        elif (self.sampling_policy == "per"):
            priorities = [e.priority for e in self.experience_buffer]
            sample = random.choices(self.experience_buffer, k=n, weights=priorities)
        return sample

    def update_batch_priorities(self, batch, td_error) -> None:

        """
        Implementation of Prioritized Replay Memory.

        Parameters:
        - batch: The batch of Experiences whose priorities have to be
          updated.
        - td_error: The temporal difference error of each experience
          of the batch, used to calc the priority.
        """

        if (self.sampling_policy != "per"): return
        denominator = [e.priority for e in self.experience_buffer]
        denominator = torch.tensor(denominator,
                                   dtype=torch.float32).to(self.device)
        denominator = denominator + self.epsilon
        denominator = torch.pow(denominator, self.alpha)
        denominator = torch.sum(denominator)
        numerator = td_error + self.epsilon
        numerator = torch.pow(numerator, self.alpha)
        new_priorities = numerator / denominator
        for i, new_p in enumerate(new_priorities):
            batch[i].priority = new_p.item()

if __name__ == "__main__":
    print("Testing the experience sampler")
    my_exp_sampler = ExperienceSampler(10, sampling_policy = "simple")

    for i in range(10):
        experience = Experience([0,1,2,3,4,5,6,7,8,9],
                                [i],
                                [10],
                                [1+i,2+i,3+i,4+i,5+i,6+i,7+i,8+i,9+i,10+i],
                                False,
                                8)
        my_exp_sampler.add_experience(experience)
        sampled_experience = my_exp_sampler.sample_experience(n=1)[0]
        sampled_experience.priority = 10
        print("Sampled Experience:")
        print(sampled_experience)
    print("Check buffer:")
    [print(experience) for experience in my_exp_sampler.experience_buffer]
