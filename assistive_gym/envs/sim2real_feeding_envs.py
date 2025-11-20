from .sim2real_feeding import Sim2RealFeedingEnv
from .agents import mico, human
from .agents.mico import Mico

from .agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'right'
human_controllable_joint_indices = human.head_joints

class Sim2RealFeedingMicoEnv(Sim2RealFeedingEnv):
    def __init__(self):
        super(Sim2RealFeedingMicoEnv, self).__init__(robot=Mico(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class Sim2RealFeedingMicoHumanEnv(Sim2RealFeedingEnv, MultiAgentEnv):
    def __init__(self):
        super(Sim2RealFeedingMicoHumanEnv, self).__init__(robot=Mico(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:Sim2RealFeedingMicoHuman-v1', lambda config: Sim2RealFeedingMicoHumanEnv())