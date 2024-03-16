import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import plot_learning_curve, get_learning_curve_data

rnn_df = pd.read_csv('project/code/model/rnn/city_48/test_9.1/budget_40000/starting_city_0/total_ep_5010/num_test_0/training_data_each_episode.csv')



episode = rnn_df['episode']
rnn_training_reward = rnn_df['reward']
# # loss = rnn_df['avg_loss']
# rnn_avg_loss = get_learning_curve_data(episode, loss)
# rnn_training_time = rnn_df['execution_time']
rnn_running_avg_data = get_learning_curve_data(episode, rnn_training_reward)
# rnn_training_time_avg = get_learning_curve_data(episode, rnn_training_time)
# plt.plot(rnn_running_avg_data[0], rnn_running_avg_data[1])
# plt.show()

# plt.plot(rnn_avg_loss[0],rnn_avg_loss[1])
# plt.show()