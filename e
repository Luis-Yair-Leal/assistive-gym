[1mdiff --git a/assistive_gym/envs/object_handover.py b/assistive_gym/envs/object_handover.py[m
[1mindex 448d913..5fe58cc 100644[m
[1m--- a/assistive_gym/envs/object_handover.py[m
[1m+++ b/assistive_gym/envs/object_handover.py[m
[36m@@ -86,7 +86,6 @@[m [mclass ObjectHandoverEnv(AssistiveEnv):[m
             # Co-optimization with both human and robot controllable[m
             return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}[m
     [m
[31m-[m
     def get_total_force(self):[m
         total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])[m
         tool_force = np.sum(self.tool.get_contact_points()[-1])[m
[1mdiff --git a/assistive_gym/envs/sim2real_feeding.py b/assistive_gym/envs/sim2real_feeding.py[m
[1mindex 20dd3bc..14d771b 100644[m
[1m--- a/assistive_gym/envs/sim2real_feeding.py[m
[1m+++ b/assistive_gym/envs/sim2real_feeding.py[m
[36m@@ -30,7 +30,7 @@[m [mclass Sim2RealFeedingEnv(AssistiveEnv):[m
         reward_action = -np.linalg.norm(action) # Penalize actions[m
 [m
         # Total reward is composed by distance mouth target, action, food in the spoon and extra preferences[m
[31m-        reward = self.config('distance_weight')*(-reward_distance_mouth_target) + self.config('action_weight')*reward_action + self.config('food_reward_weight')*reward_food + preferences_score[m
[32m+[m[32m        reward = self.config('distance_weight')*(reward_distance_mouth_target) + self.config('action_weight')*reward_action + self.config('food_reward_weight')*reward_food + preferences_score[m
         # print(self.config('distance_weight')*reward_distance_mouth_target, self.config('action_weight')*reward_action, self.config('food_reward_weight')*reward_food, preferences_score)[m
 [m
         if self.gui and reward_food != 0:[m
[36m@@ -166,6 +166,7 @@[m [mclass Sim2RealFeedingEnv(AssistiveEnv):[m
         return self._get_obs()[m
 [m
     '''[m
[32m+[m[32m    Past Reward Functtion (without Sim2Real Considerations)[m
     def get_food_rewards(self):[m
         Check all food particles to see if they have left the spoon or entered the person's mouth[m
         Give the robot a reward or penalty depending on food particle status[m
[36m@@ -210,15 +211,14 @@[m [mclass Sim2RealFeedingEnv(AssistiveEnv):[m
         foods_to_remove = [][m
         foods_active_to_remove = [][m
 [m
[31m-        # Position and orientation of the spoon[m
[31m-        spoon_pos, spoon_orient = self.tool.get_base_pos_orient() [m
[32m+[m[32m        spoon_pos, spoon_orient = self.tool.get_base_pos_orient() # Position and orientation of the spoon[m
         spoon_pos_real, spoon_orient_real = self.robot.convert_to_realworld(spoon_pos, spoon_orient)  # Convert the relative position and orientation to global position and orientation[m
         roll, pitch, yaw = p.getEulerFromQuaternion(spoon_orient_real) # Convert quaternion to euler[m
 [m
[31m-        # Penalty for tilt[m
[31m-        tilt_penalty = abs(roll) + abs(pitch)  # Negative reward for tilt[m
[32m+[m[32m        # Reward related of tilt[m
[32m+[m[32m        tilt_penalty = abs(roll) + abs(pitch)  # Penalty for tilt[m
 [m
[31m-        # Penalty for change orientation[m
[32m+[m[32m        # Reward related to change orientation[m
         if self.prev_spoon_orient is not None:[m
             prev_roll, prev_pitch, _ = p.getEulerFromQuaternion(self.prev_spoon_orient)[m
             angular_change = abs(roll - prev_roll) + abs(pitch - prev_pitch)[m
[36m@@ -227,7 +227,7 @@[m [mclass Sim2RealFeedingEnv(AssistiveEnv):[m
         self.prev_spoon_orient = spoon_orient_real[m
 [m
         # Reward for reaching the target[m
[31m-        distance_to_mouth = np.linalg.norm(self.target_pos - spoon_pos_real)[m
[32m+[m[32m        distance_to_mouth = -np.linalg.norm(self.target_pos - spoon_pos_real)[m
 [m
         if distance_to_mouth < 0.03:[m
             food_reward = 100[m
[36m@@ -250,7 +250,7 @@[m [mclass Sim2RealFeedingEnv(AssistiveEnv):[m
         self.foods = [f for f in self.foods if f not in foods_to_remove][m
         self.foods_active = [f for f in self.foods_active if f not in foods_active_to_remove][m
 [m
[31m-        reward = - 1.0 * tilt_penalty - 0.25 * food_velocity + 0.01 * reward_force_nontarget # - 0.5 * angular_change[m
[32m+[m[32m        reward =  - 0.25 * food_velocity + 0.01 * reward_force_nontarget # - 5.0 * tilt_penalty - 0.5 * angular_change[m
 [m
         return food_reward, reward, distance_to_mouth[m
 [m
[1mdiff --git a/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000040/checkpoint-40 b/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000040/checkpoint-40[m
[1mindex f44dd23..47cf272 100644[m
Binary files a/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000040/checkpoint-40 and b/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000040/checkpoint-40 differ
[1mdiff --git a/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000040/checkpoint-40.tune_metadata b/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000040/checkpoint-40.tune_metadata[m
[1mindex 4d33389..f1f1ca0 100644[m
Binary files a/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000040/checkpoint-40.tune_metadata and b/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000040/checkpoint-40.tune_metadata differ
[1mdiff --git a/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000041/.is_checkpoint b/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000041/.is_checkpoint[m
[1mdeleted file mode 100644[m
[1mindex e69de29..0000000[m
[1mdiff --git a/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000041/checkpoint-41 b/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000041/checkpoint-41[m
[1mdeleted file mode 100644[m
[1mindex 8b59298..0000000[m
Binary files a/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000041/checkpoint-41 and /dev/null differ
[1mdiff --git a/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000041/checkpoint-41.tune_metadata b/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000041/checkpoint-41.tune_metadata[m
[1mdeleted file mode 100644[m
[1mindex 9102eff..0000000[m
Binary files a/trained_models/ppo/Sim2RealFeedingMico-v1/checkpoint_000041/checkpoint-41.tune_metadata and /dev/null differ
