"""
    Generates a stream of gameplay for a given agent.

    A folder 'stream' is created whose subfolders contain all the states, visually displayed frames, Q-values,
    saliency maps and features (output of the second to last layer).

    At the very end *overlay_stream* is used to overlay each frame with a saliency map.
    This can also be redone later using *overlay_stream* to save time while trying different overlay styles.
"""


import gym
import matplotlib.pyplot as plt
import numpy as np
import keras
#from argmax_analyzer import Argmax
#import overlay_stream
#from gym_minigrid.wrappers import *
#import gym s_maze
import pandas as pd
import tensorwatch as tw
import sys
import scipy
import seaborn as sns
from matplotlib.colors import ListedColormap
from collections import OrderedDict
import os
# from varname import nameof

import chainer
import h5py

class DataVault:
    #dictionaries for storing all the info which will be shoved into dataframes later
    main_data_dict = OrderedDict()
    per_episode_action_distribution_dict = {}
    df_list = []
    
    #keep track of steps
    step = 1

    #def __init__(self):
        
    def print_df(self, stats_df):
        print("DF: ")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(stats_df)
            
    def df_to_csv(self, stream_directory):
        counter = 1
        for df in self.df_list:
            filename = "df" + str(counter) + ".csv"
            filepath = os.path.join(stream_directory, filename)
            df.to_csv(filepath, index=True)
            print("Output to csv")
            counter = counter + 1
        
    def stack_counts_of_actions_per_episode(self, episode, action_episode_sums):
        episode_sums = { episode: {
            'action 0 episode sum': action_episode_sums[0],
            'action 1 episode sum': action_episode_sums[1],
            'action 2 episode sum': action_episode_sums[2],
#            'action 3 episode sum': action_episode_sums[3],
#            'action 4 episode sum': action_episode_sums[4],
#            'action 5 episode sum': action_episode_sums[4],
#            'action 6 episode sum': action_episode_sums[6],
#            'action 7 episode sum': action_episode_sums[7],
#            'action 8 episode sum': action_episode_sums[8],
                }
            }
        self.per_episode_action_distribution_dict.update(episode_sums)
        print(self.per_episode_action_distribution_dict)
        
    def store_data(self, action, action_name, action_episode_sums, action_total_sums, reward, done, info, lives):
        #need to find some other way to store: Observations, argmax, features, q value
        
        end_of_episode = False
        end_of_epoch = False
        
#        print("Lives: " + str(lives))
        # If dataframe is not empty...
        if len(self.main_data_dict) != 0:
            lastElem = list(self.main_data_dict.keys())[-1]
#            print("Last element is: " + str(lastElem))
            
            last_lives_left = self.main_data_dict[lastElem]['lives']
            episode = self.main_data_dict[lastElem]['episode']
            epoch = self.main_data_dict[lastElem]['epoch']
            episode_reward = self.main_data_dict[lastElem]['episode reward'] + reward
            epoch_reward = self.main_data_dict[lastElem]['epoch reward'] + reward
            total_reward = self.main_data_dict[lastElem]['total reward'] + reward
            self.step = self.step + 1
            episode = self.main_data_dict[lastElem]['episode']
            epoch = self.main_data_dict[lastElem]['epoch']
            episode_step = self.main_data_dict[lastElem]['episode step'] + 1
            epoch_step = self.main_data_dict[lastElem]['epoch step'] + 1
        else:
#            print("Should be setting up dict for the first time")
            last_lives_left = lives
            episode = epoch = episode_step = epoch_step = self.step
            episode_reward = epoch_reward = total_reward = reward
#        print("Episode is; " + str(episode))
            
            
        eoe_flag = False
        #first check if new episode or new epoch started
        if (lives != last_lives_left):
            eoe_flag = True
            # Add to other dictionary, for stacked bar chart, as currently set
#            print("Added to stacked chart dictionary")
            self.stack_counts_of_actions_per_episode(episode, action_episode_sums)
            episode_reward = reward
            episode = episode + 1
            episode_step = 1
            end_of_episode = True
            for x in range(len(action_episode_sums)):
                action_episode_sums[x] = 0
            
            if (done):
            # Have used up all three lives, therfore an "epoch" is over, and need to zero out accumulators
                epoch_reward = reward
                epoch = epoch + 1
                end_of_epoch = True
                epoch_step = 1
#                print("end of episode and epoch is true ")
                eoe_flag = True
        
        # Up correct action sum
        temp_action_episode_sum = action_episode_sums[action]
        action_episode_sums[action] = temp_action_episode_sum + 1
        
        temp_action_total_sum = action_total_sums[action]
        action_total_sums[action] = temp_action_total_sum + 1
#        print("end of episode and epoch is: ")
#        print(end_of_episode)
#        print(end_of_epoch)

        step_stats = { self.step: {
            'action_name': action_name,
            'action': action,
            'reward': reward,
            'episode reward': episode_reward,
            'epoch reward': epoch_reward,
            'total reward': total_reward,
            'lives': lives,
            'end of episode': end_of_episode,
            'end of epoch': end_of_epoch,
            'info': info,
            'episode': episode,
            'episode step': episode_step,
            'epoch': epoch,
            'epoch step': epoch_step,
            'step': self.step}
            }
            
        # add carefully the action sums to the dictionary
        for action_number in range(len(action_episode_sums)):
#            print("Action in list: ")
#            print(action_number)
            index_name = "action " + str(action_number) + " episode sum"
            step_stats[self.step][index_name] = action_episode_sums[action_number]
            index_name = "action " + str(action_number) + " total sum"
            step_stats[self.step][index_name] = action_total_sums[action_number]
        
#        print("Adding to main dict")
        self.main_data_dict.update(step_stats)
        
        return (action_episode_sums, action_total_sums)

    def make_dataframes(self):
        main_df = pd.DataFrame.from_dict(self.main_data_dict, orient='index')
        stacked_bar_df = pd.DataFrame.from_dict(self.per_episode_action_distribution_dict, orient='index')
        self.df_list.append(main_df)
        self.df_list.append(stacked_bar_df)
