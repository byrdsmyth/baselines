import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon = 0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        return 0
        
class AltObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(AltObservationWrapper, self).__init__(env)
        print("Observations: ")
        print(self.observation_space)
        return None
        
    def observation(self, observation):
        return None

class AltRewardsWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(AltRewardsWrapper, self).__init__(env)
        return None
        
    def reward(self, reward):
        if reward > 10:
            print("Reward received: ")
            print(reward)
        return reward
        
class AltFreewayRewardsWrapper(gym.RewardWrapper):
# chicken gets one point for getting across the highway
# in normal game, set to a higher number to encourage
# safely getting across the road
    def __init__(self, env):
        super(AltFreewayRewardsWrapper, self).__init__(env)
        return None
        
    def reward(self, reward):
        if reward > 0:
            print("Reward received: ")
            print(reward)
            reward = 10000
        return reward
        
class PacmanCherriesRewardsWrapper(gym.RewardWrapper):
# chicken gets one point for getting across the highway
# in normal game, set to a higher number to encourage
# safely getting across the road
    def __init__(self, env):
        super(PacmanCherriesRewardsWrapper, self).__init__(env)
        return None
        
    def reward(self, reward):
        if reward == 100:
            print("Reward received: ")
            print(reward)
            reward = 10000
        else:
            reward = 0
        return reward
        
class PacmanClearTheBoardRewardsWrapper(gym.RewardWrapper):
# chicken gets one point for getting across the highway
# in normal game, set to a higher number to encourage
# safely getting across the road
    def __init__(self, env):
        super(PacmanClearTheBoardRewardsWrapper, self).__init__(env)
        return None
        
    def reward(self, reward):
        if reward == 10:
            print("Reward received: ")
            print(reward)
            reward = 1000
        elif reward == 0:
            reward = -10
#        elif reward > 0:
#            reward = 0
        return reward

class pacmanFourActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(pacmanFourActionsWrapper, self).__init__(env)
        return None
    
    def action(self, act):
        # ['NOOP' = 0, 'UP' = 1, 'RIGHT' = 2, 'LEFT' = 3, 'DOWN' = 4, 'UPRIGHT' = 5, 'UPLEFT' = 6, 'DOWNRIGHT' = 7, 'DOWNLEFT' = 8]
        if act == 5:
            act_choices = [1,2]
            newAct = sample(act_choices,  1)
            print("Now action is: ")
            print(newAct)
            return newAct
        elif act == 6:
            act_choices = [1,3]
            newAct = sample(act_choices,  1)
            print("Now action is: ")
            print(newAct)
            return newAct
        elif act == 7:
            act_choices = [4,2]
            newAct = sample(act_choices,  1)
            print("Now action is: ")
            print(newAct)
            return newAct
        elif act == 8:
            act_choices = [4,3]
            newAct = sample(act_choices,  1)
            print("Now action is: ")
            print(newAct)
            return newAct
        return act
        
class FearDeathWrapper(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        # Add sharp negative reward to encourage fear
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = -10
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class FreewayUpRewarded(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        if action == 1:
            reward = 1
        elif action == 2:
            reward = -0.5
        return obs, reward, done, info
