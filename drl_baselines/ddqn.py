"""Double DQN comparison algorithm."""

from .dqn import DQNBaseline


class DDQNBaseline(DQNBaseline):
    """Double DQN baseline using online action selection and target evaluation."""

    algorithm_name = "DDQN"
    double_q = True
