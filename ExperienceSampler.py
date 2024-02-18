from collections import deque
import random
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
    
    def __init__(self, buffer_max_size, sampling_policy: str = "simple") -> None:
    
        self.experience_buffer: deque[Experience] =\
            deque([], maxlen=buffer_max_size)
        self.sampling_policy = sampling_policy
        if (self.sampling_policy == "per"):
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
            pass
        return sample

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
        sampled_experience = my_exp_sampler.sample_experience(n=1)
        print("Sampled Experience:")
        [print(experience) for experienced in sampled_experience]
    print("Sampling 5 experiences:")
    sampled_experiences = my_exp_sampler.sample_experience(n=5)
    print(sampled_experiences)
    [print(experience) for experience in sampled_experiences]
    

