from collections import deque
import random
import torch
from .Experience import Experience


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

    def add_experience(self, experience: Experience):
        """Store an experience into the buffer.
        """
        self.experience_buffer.append(experience)

    def sample_experience(self, n: int = 1) -> list:
        """Return a list of experiences sampled according to the
        sampling policy.
        """
        if (self.sampling_policy == "simple"):
            sample = random.sample(self.experience_buffer, n)
        elif (self.sampling_policy == "per"):
            priorities = [e.priority for e in self.experience_buffer]
            sample = random.choices(self.experience_buffer,
                                    k=n,
                                    weights=priorities)
        return sample

    def update_batch_priorities(self,
                                batch: list,
                                td_error:torch.tensor) -> None:
        """Implementation of Prioritized Replay Memory.

        Args:
            batch (_type_): _description_
            td_error (_type_): _description_
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

