import numpy as np
import pybullet as p
import csv #
import os #
from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture

# Sim to Real Edition of Feeding Task focused on disable people
class Sim2RealFeedingEnv(AssistiveEnv):
    def __init__(self, robot='mico', human=False):
        # Robot type, human,, task, frame_skipe,  time_step, observations of the robot (18 + joints - wheel joints), human observations (19 + body joints)
        super(Sim2RealFeedingEnv, self).__init__(robot=robot, human=human, task='feeding', frame_skip=5, time_step=0.02, obs_robot_len=(18 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(19 + len(human.controllable_joint_indices)))


    def step(self, action): # Take step given an action
        '''In the step function, we have to consider the actions made by the agent in a unique time step and the output must be the observations after taking the actions by the agent,
        the reward and the 'done' flag that indicates if the task has been solved succesfully. We have to change some rewards in order to take only rewards that depends on the observations.'''
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient() #  Get the spoon position and orientation
        self.prev_spoon_orient = np.array(spoon_orient) # Get the previous spoon orientation to compare
        spoon_orient_euler = p.getEulerFromQuaternion(spoon_orient) # Get euler angles
        
        if self.human.controllable: # If we use colaborative control
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action, action_multiplier = 0.05) # Take the actions that agent must do, and traduce it to joint movements for a time step.

        obs = self._get_obs() # Get the observation after the action

        # Get human preferences Editar para que vaya dentro de la función de recompensa
        reward_food, preferences_score, reward_distance_mouth_target = self.get_food_rewards() # Takes the rewards, the velocities list, and special reward for hitting person

        reward_action = -np.linalg.norm(action) # Penalize actions

        # Total reward is composed by distance mouth target, action, food in the spoon and extra preferences
        reward = self.config('distance_weight')*(-reward_distance_mouth_target) + self.config('action_weight')*reward_action + self.config('food_reward_weight')*reward_food + preferences_score
        # print(self.config('distance_weight')*reward_distance_mouth_target, self.config('action_weight')*reward_action, self.config('food_reward_weight')*reward_food, preferences_score)
        
        ##############################################################
        # Get the joint states of the robot
        _, motor_positions, _, _ = self.robot.get_motor_joint_states()
        print("\n=== Data from the environment ===")
        print("\nTarget position", self.target_pos)
        print("\nSpoon position", spoon_pos)
        print("\nSpoon orient", spoon_orient_euler)
        print("\nJoint positions:", motor_positions)

        nombre_archivo = 'trayectoria_robot.csv'
        
        # Comprobamos si el archivo ya existe para saber si escribir la cabecera
        existe = os.path.isfile(nombre_archivo)

        # Usamos 'a' (append) para que NO se borre lo anterior
        with open(nombre_archivo, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Solo escribimos la cabecera la primera vez que se crea el archivo
            if not existe:
                writer.writerow(['tx', 'ty', 'tz', 'sx', 'sy', 'sz', 's_roll', 's_pitch', 's_yaw'])

            # Guardamos los datos (asegúrate de que todos sean valores simples)
            writer.writerow([*self.target_pos, *spoon_pos, *spoon_orient_euler])
        ##############################################################

        if self.gui and reward_food != 0:
            print('Task success:', self.task_success, 'Food reward:', reward_food)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.total_food_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}


    def _get_obs(self, agent=None):
        '''The _get_obs method indicates the observations obtained by the environment, precisely, the spoon actual position and orientation, the target position, the robot joint angles,
        the head position and orientation, and the spoon force.
        Observations taken in consideration:
        - Spoon real pose and orientation 
        - Target pose and orientation
        - Robot joint angles
        - Difference on real spoon and target spoon position
        - Spoon force on human
        '''
        # Position and orientation of the spoon
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient() 
        spoon_pos_real, spoon_orient_real = self.robot.convert_to_realworld(spoon_pos, spoon_orient)  # Convert the relative position and orientation to global position and orientation

        # Joint angles
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices) 
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi # Fix joint angles to be in [-pi, pi]

        # Target position
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)  # Local position and orientation of the head
        head_pos_real, head_orient_real = self.robot.convert_to_realworld(head_pos, head_orient) # Global position and orientation of the head
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos) # Global target position (mouth)
        diff_spoon_target = spoon_pos_real - target_pos_real # Difference between spoon and target

        # Force applied to the human by the spoon
        self.robot_force_on_human, self.spoon_force_on_human = self.get_total_force() 
        self.total_force_on_human = self.robot_force_on_human + self.spoon_force_on_human # Total force applied to the human

        # OBSERVATIONS OF THE RL MODEL
        robot_obs = np.concatenate([spoon_pos_real, spoon_orient_real, spoon_pos_real - target_pos_real, robot_joint_angles, head_pos_real, head_orient_real, [self.spoon_force_on_human]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            spoon_pos_human, spoon_orient_human = self.human.convert_to_realworld(spoon_pos, spoon_orient) 
            head_pos_human, head_orient_human = self.human.convert_to_realworld(head_pos, head_orient)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate([spoon_pos_human, spoon_orient_human, diff_spoon_target, human_joint_angles, head_pos_human, head_orient_human, [self.robot_force_on_human, self.spoon_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs


    def reset(self):
        super(Sim2RealFeedingEnv, self).reset()
        self.build_assistive_env('wheelchair')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        # Update robot and human motor gains
        self.robot.motor_gains = self.human.motor_gains = 0.025

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        # Create a table
        self.table = Furniture()
        self.table.init('table', self.directory, self.id, self.np_random)

        self.generate_target()

        #p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        p.resetDebugVisualizerCamera(cameraDistance=1.50, cameraYaw=-40, cameraPitch=-45, cameraTargetPosition=[0.2, 0, 0.75], physicsClientId=self.id)

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.08]*3)

        target_ee_pos = np.array([-0.15, -0.65, 1.15]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], arm='right', tools=[self.tool], collision_objects=[self.human, self.table, self.furniture])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        # Place a bowl on a table
        self.bowl = Furniture()
        self.bowl.init('bowl', self.directory, self.id, self.np_random)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Generate food
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()
        self.prev_spoon_orient = np.array(spoon_orient)
        food_radius = 0.005
        food_mass = 0.001
        batch_positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    batch_positions.append(np.array([i*2*food_radius-0.005, j*2*food_radius, k*2*food_radius+0.01]) + spoon_pos)
        self.foods = self.create_spheres(radius=food_radius, mass=food_mass, batch_positions=batch_positions, visual=False, collision=True)
        colors = [[60./256., 186./256., 84./256., 1], [244./256., 194./256., 13./256., 1],
                  [219./256., 50./256., 54./256., 1], [72./256., 133./256., 237./256., 1]]
        for i, f in enumerate(self.foods):
            p.changeVisualShape(f.body, -1, rgbaColor=colors[i%len(colors)], physicsClientId=self.id)
        self.total_food_count = len(self.foods)
        self.foods_active = [f for f in self.foods]

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop food in the spoon
        for _ in range(25):
            p.stepSimulation(physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    '''
    Past Reward Function (without Sim2Real Considerations)
    def get_food_rewards(self):
        Check all food particles to see if they have left the spoon or entered the person's mouth
        Give the robot a reward or penalty depending on food particle status
        
        food_reward = 0
        food_hit_human_reward = 0
        food_mouth_velocities = []
        foods_to_remove = []
        foods_active_to_remove = []
        for f in self.foods:   #Food particles on the spoon
            food_pos, food_orient = f.get_base_pos_orient()
            distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
            if distance_to_mouth < 0.03:
                # Food is close to the person's mouth. Delete particle and give robot a reward
                food_reward += 20
                self.task_success += 1
                food_velocity = np.linalg.norm(f.get_velocity(f.base))
                food_mouth_velocities.append(food_velocity)
                foods_to_remove.append(f)
                foods_active_to_remove.append(f)
                f.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                continue
            elif len(f.get_closest_points(self.tool, distance=0.1)[-1]) == 0:
                # Delete particle and give robot a penalty for spilling food
                food_reward -= 5
                foods_to_remove.append(f)
                continue
        for f in self.foods_active:
            if len(f.get_contact_points(self.human)[-1]) > 0:
                # Record that this food particle just hit the person, so that we can penalize the robot
                food_hit_human_reward -= 1
                foods_active_to_remove.append(f)
        self.foods = [f for f in self.foods if f not in foods_to_remove]
        self.foods_active = [f for f in self.foods_active if f not in foods_active_to_remove]
        return food_reward, food_mouth_velocities, food_hit_human_reward
        '''

    def get_food_rewards(self):
        '''Reward function using spoon position, orientation and contact events, while still keeping food particles for visualization.'''
        food_reward = 0
        food_hit_human_reward = 0
        foods_to_remove = []
        foods_active_to_remove = []

        # ---------------------------------
        # Spoon Reward
        # ---------------------------------
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient() # Position and orientation of the spoon
        spoon_pos_real, spoon_orient_real = self.robot.convert_to_realworld(spoon_pos, spoon_orient)  # Convert the relative position and orientation to global position and orientation
        #roll, pitch, yaw = p.getEulerFromQuaternion(spoon_orient_real) # Convert spoon orientation to euler angles
        #distance_to_mouth = np.linalg.norm(self.target_pos - spoon_pos_real) # Reward for reaching the target
        roll, pitch, yaw = p.getEulerFromQuaternion(spoon_orient) # Convert spoon orientation to euler angles
        distance_to_mouth = np.linalg.norm(self.target_pos - spoon_pos) # Reward for reaching the target

        # ---------------------------------
        # Reward related of tilt
        # ---------------------------------
        tilt_penalty = abs(yaw) + abs(pitch) # Penalty for tilt

        # ---------------------------------
        # Reward related to change orientation (smooth motion)
        # ---------------------------------
        if self.prev_spoon_orient is not None:
            prev_roll, prev_pitch, _ = p.getEulerFromQuaternion(self.prev_spoon_orient)
            angular_change = abs(roll - prev_roll) + abs(pitch - prev_pitch)
        else:
            angular_change = 0
        self.prev_spoon_orient = spoon_orient_real

        # ---------------------------------
        # Reward for food velocity
        # ---------------------------------
        food_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))

        # ---------------------------------
        # Penalty for touching the human
        # ---------------------------------
        reward_force_nontarget = -self.total_force_on_human

        # ---------------------------------
        # Tasks success
        # ---------------------------------
        if distance_to_mouth < 0.1:
            for f in self.foods:
                food_pos, food_orient = f.get_base_pos_orient()
                distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
                if distance_to_mouth < 0.03:
                    f.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
            food_reward = 10
            self.task_success += 1


        # ---------------------------------
        # Reward of preferences score
        # ---------------------------------
        reward = - 1.0 * tilt_penalty - 0.25 * food_velocity + 0.1 * reward_force_nontarget - 0.25 * angular_change

        return food_reward, reward, distance_to_mouth


    def get_total_force(self):
        '''Get the robot force applied on human and the force spoon force applied on human.'''
        robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        spoon_force_on_human = np.sum(self.tool.get_contact_points(self.human)[-1])
        return robot_force_on_human, spoon_force_on_human


    def generate_target(self):
        '''Set the target point on the mouth.'''
        self.mouth_pos = [0, -0.11, 0.03] if self.human.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()


    def update_targets(self):
        '''Update the targets positons and orientations.'''
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])