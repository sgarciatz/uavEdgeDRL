import unittest
from src.ExperienceSampler import ExperienceSampler
from src.Experience import Experience


class test_ExperienceSampler(unittest.TestCase):


    def test_simple_ExperienceSampler_initialization(self):
        """Check that the buffer is initialized with the correct
        parameters.
        """

        buffer_max_size = 1024
        sampling_policy = "simple"
        device = "cpu"
        epsilon = 1e-4
        alpha = 1.0
        experienceSampler = ExperienceSampler(buffer_max_size,
                                              sampling_policy,
                                              device)
        self.assertEqual(buffer_max_size,
                         experienceSampler.experience_buffer.maxlen)
        self.assertEqual(device,
                         experienceSampler.device)
        self.assertEqual(sampling_policy,
                         experienceSampler.sampling_policy)

    def test_add_experience(self):
        """Check that experiences are added to the buffer.
        """
        buffer_max_size = 1024
        sampling_policy = "simple"
        device = "cpu"
        epsilon = 1e-4
        alpha = 1.0
        experienceSampler = ExperienceSampler(buffer_max_size,
                                              sampling_policy,
                                              device,
                                              epsilon,
                                              alpha)


if __name__ == '__main__':
    unittest.main()