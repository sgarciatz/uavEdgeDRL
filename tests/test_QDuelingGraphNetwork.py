import unittest
import torch
import context

class TestQDuelingGraphNetwork(unittest.TestCase):



    def test(self):

        """Create a QDuelingGraphNetwork and perform a conv"""

        torch.manual_seed(0)
        n_uavs = 16
        n_mss = 4
        n_obs = n_uavs * ( n_mss + 2) + n_mss * 3
        layers = [128, [128, 128], 128]
        net = context.src.QDuelingGraphNetwork.QDuelingGraphNetwork(n_obs,
                                   n_uavs,
                                   n_mss,
                                   layers,
                                   "cpu")
        x = [1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1]
        x = torch.tensor(x, dtype=torch.float32).repeat(1,1)
        y = net.apply_conv(x)
        y_target = [50., 45., 45., 50., 50., 45., 45., 50., 50.,
                    45., 45., 50., 50., 45., 45., 50., 45., 38.,
                    38., 45., 45., 38., 38., 45., 45., 38., 38.,
                    45., 45., 38., 38., 45., 45., 38., 38., 45.,
                    45., 38., 38., 45., 45., 38., 38., 45., 45.,
                    38., 38., 45., 50., 45., 45., 50., 50., 45.,
                    45., 50., 50., 45., 45., 50., 50., 45., 45.,
                    50.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                     1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                     1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                     1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                     1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]
        y_target = torch.tensor(y_target, dtype=torch.float32).repeat(1,1)
        self.assertTrue(torch.all(torch.eq(y, y_target)))

unittest.main()


