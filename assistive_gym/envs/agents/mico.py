import os
import numpy as np
import pybullet as p
from .robot import Robot

class Mico(Robot):
    def __init__(self, controllable_joints='right'):
        right_arm_joint_indices = [2, 3, 4, 5, 6, 7] # Controllable arm joints
        left_arm_joint_indices = right_arm_joint_indices # Controllable arm joints
        wheel_joint_indices = []
        right_end_effector = 8 # Used to get the pose of the end effector
        left_end_effector = right_end_effector # Used to get the pose of the end effector
        right_gripper_indices = [9, 11, 13] # Gripper actuated joints
        left_gripper_indices = right_gripper_indices # Gripper actuated joints
        right_tool_joint = 8 # Joint that tools are attached to
        left_tool_joint = right_tool_joint # Joint that tools are attached to
        right_gripper_collision_indices = list(range(7, 15)) # Used to disable collision between gripper and tools
        left_gripper_collision_indices = right_gripper_collision_indices # Used to disable collision between gripper and tools
        gripper_pos = {'scratch_itch': [1]*3, # Gripper open position for holding tools
                       'feeding': [1.33]*3,
                       'drinking': [0.63]*3,
                       'bed_bathing': [1.1]*3,
                       'dressing': [1.33]*3,
                       'arm_manipulation': [1.05]*3}
        tool_pos_offset = {'scratch_itch': [0, 0, 0.02], # Position offset between tool and robot tool joint
                           'feeding': [0.0, 0.0, 0.1], # Original offset between spoon and end effector [0.1, -0.0225, 0.03]
                           'drinking': [0.05, -0.005, 0],
                           'bed_bathing': [-0.01, 0, 0.03],
                           'arm_manipulation': [0.075, 0, 0.14]}
        tool_orient_offset = {'scratch_itch': [0, -np.pi/2.0, 0], # RPY orientation offset between tool and robot tool joint
                              'feeding': [0, np.pi, 0], # Original spoon orientation [-0.1, np.pi/2.0, 0.0],
                              'drinking': [0, -np.pi/2.0, np.pi/2.0],
                              'bed_bathing': [0, -np.pi/2.0, 0],
                              'arm_manipulation': [np.pi/2.0, -np.pi/2.0, 0]}
        
        pos = [0.25, -0.65, 0.82] # wheelchair position, [-0.35, -0.3, 0.3] previous accepted position. Previous height [0.25, -0.65, 0.7]
        toc_base_pos_offset = {'scratch_itch': [-0.35, -0.3, 0.3], # Robot base offset before TOC base pose optimization
                               'feeding': pos,
                               'drinking': [-0.35, -0.3, 0.3],
                               'bed_bathing': [-0.05, 1.05, 0.6],
                               'dressing': [0.35, -0.3, 0.3],
                               'arm_manipulation': [-0.25, 1.15, 0.6]}
        toc_ee_orient_rpy = {'scratch_itch': [0, np.pi/2.0, 0], # Initial end effector orientation
                             'feeding': [np.pi/2.0, 0, np.pi],  # Original end effector orientation for feeding [np.pi/2.0, 0, np.pi/2.0],
                             'drinking': [0, np.pi/2.0, 0],
                             'bed_bathing': [0, np.pi/2.0, 0],
                             'dressing': [[0, -np.pi/2.0, 0]],
                             'arm_manipulation': [0, np.pi/2.0, 0]}
        wheelchair_mounted = True

        super(Mico, self).__init__(controllable_joints, right_arm_joint_indices, left_arm_joint_indices, wheel_joint_indices, right_end_effector, left_end_effector, right_gripper_indices, left_gripper_indices, gripper_pos, right_tool_joint, left_tool_joint, tool_pos_offset, tool_orient_offset, right_gripper_collision_indices, left_gripper_collision_indices, toc_base_pos_offset, toc_ee_orient_rpy, wheelchair_mounted, half_range=False)

    def init(self, directory, id, np_random, fixed_base=True):
        self.body = p.loadURDF(os.path.join(directory, 'mico', 'm1n6s300_gym.urdf'), useFixedBase=fixed_base, basePosition=[-1, -1, 0.5], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=id)
        super(Mico, self).init(self.body, id, np_random)
