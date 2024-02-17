import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class TrainLogger(object):


    """
    TrainLogger is responsable of outputting information about the 
    training process in a human-readable format.
    """

    def __init__(self):

        """
        Constructor of the TrainLogger object, it creates the DataFrame
        that holds all the information
        """

        self.writer = SummaryWriter()
        columns = ["Training Step",
                   "Exploration Rate",
                   "Loss",
                   "Avg episode reward",
                   "Avg episode length"]
        self.data = []


    def add_training_step(self, step, expl_rate, loss, reward, ep_length):

        """
        Inserts a new row into the data DataFrame with the step
        information.

        Arguments:
        - step: The number of the current step.
        - loss: The mean loss of the training step.
        - expl_rate: The current exploration rate.
        - reward: The reward obtained in the validation.
        - ep_length: The mean episode duration of the validation.
        """

        row = {"Training Step": step,
               "Exploration Rate": expl_rate,
               "Loss": loss,
               "Avg episode reward": reward,
               "Avg episode length": ep_length}
        self.data.append(row)

        self.writer.add_scalar("Exploration Rate / Training Step",
                          row["Exploration Rate"],
                          row["Training Step"])
        self.writer.add_scalar("Loss / Training Step",
                          row["Loss"],
                          row["Training Step"])
        self.writer.add_scalar("Avg episode reward / Training Step",
                          row["Avg episode reward"],
                          row["Training Step"])
        self.writer.add_scalar("Avg episode length / Training Step",
                          row["Avg episode length"],
                          row["Training Step"])
        self.writer.flush()

    def print_training_header(self):

        """
        Print the header of the training table.
        """

        print("╔════════╦═══════════╦════════╦═══════════╦═══════════╗")
        print("║Training║Exploration║  Loss  ║Avg Episode║Avg Episode║")
        print("║  step  ║   rate    ║        ║  Reward   ║  Lenght   ║")
        print("╠════════╬═══════════╬════════╬═══════════╬═══════════╣")

    def print_training_step(self):

        """
        Print the results of a training step.
        """
        row = self.data[-1]
        
        step_str = str(row["Training Step"]).center(8)
        loss = row["Loss"]
        loss_str = str(round(loss, 3)).center(8)
        expl_rate = row["Exploration Rate"]
        expl_str = round(expl_rate, 5)
        expl_str = str(expl_str).center(11)
        reward = row["Avg episode reward"]
        reward_str = str(reward).center(11)
        ep_length = row["Avg episode length"]
        ep_length_str = str(ep_length).center(11)
        print(f"║{step_str}║{expl_str}║{loss_str}"\
               + f"║{reward_str}║{ep_length_str}║")

    def print_training_footer(self):

        """
        Print the footer of the training table.
        """

        print("╚═════════════════════════════════════════════════════╝")

    def plot_rewards(self):

        """
        Plot the rewards obtained during training.
        """ 

        df = pd.DataFrame(self.data)
        y = df["Avg episode reward"]
        x = range(y.count())
        plt.plot(x, y)
        plt.show()
