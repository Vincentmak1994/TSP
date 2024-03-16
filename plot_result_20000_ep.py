import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import plot_learning_curve, get_learning_curve_data

budgets = [10000, 20000, 30000, 40000]
# 
for budget in budgets:
    rnn_df = pd.read_csv(f'project/code/model/rnn/city_48/test_9.2/budget_{budget}/starting_city_0/total_ep_5010/num_test_0/training_data_each_episode.csv')
    'project/code/model/rnn/city_48/test_10/budget_10000/starting_city_5/total_ep_5010/num_test_0/training_data_each_episode.csv'


    episode = rnn_df['episode']
    rnn_training_reward = rnn_df['reward']
    rnn_running_avg_data = get_learning_curve_data(episode, rnn_training_reward)
    # rnn_training_time_avg = get_learning_curve_data(episode, rnn_training_time)
    # plt.plot(rnn_training_time_avg[0], rnn_training_time_avg[1])
    # plt.show()
    # print(rnn_running_avg_data)

    marl_v2_df = pd.read_csv('project/code/model/marl_v2/test_9/ep_5000_5000/budget_10000_40000/num_agent_3_3/training/marl_10_city.csv')
    marl_v2_df = marl_v2_df[marl_v2_df['total_budget'] == budget]
    'project/code/model/marl_v2/test_8.0/ep_20000_20000/budget_2000_150000/num_agent_3_3/training/marl_10_city.csv'
    'project/code/model/marl_v2/test_7.4/ep_20000_20000/budget_2000_150000/num_agent_3_3/training/marl_10_city.csv'
    'project/code/model/marl_v2/test_7.2/ep_20000_20000/budget_6000_6000/num_agent_3_3/training/marl_10_city.csv'

    'project/code/model/marl_v2/test_3/ep_20000_20000/budget_2000_150000/num_agent_3_3/training/marl_10_city.csv'


    episode = marl_v2_df['episode']
    marl_v2_training_reward = marl_v2_df['max_reward']
    # marl_v2_training_time = marl_v2_df['execution_time']
    marl_v2_running_avg_data = get_learning_curve_data(episode, marl_v2_training_reward)
    # marl_v2__time_avg = get_learning_curve_data(episode, marl_v2_training_time)

    # plt.axhline(y=479, label='optimal', color='g', linestyle='-')

    plt.plot(marl_v2_running_avg_data[0], marl_v2_running_avg_data[1], label='P-MARL', color='r')
    plt.fill_between(marl_v2_running_avg_data[0], marl_v2_running_avg_data[1]-marl_v2_running_avg_data[2], marl_v2_running_avg_data[1]+marl_v2_running_avg_data[2], color='#FFBEAF')

    plt.plot(rnn_running_avg_data[0], rnn_running_avg_data[1], label='RNN', color='b')
    plt.fill_between(rnn_running_avg_data[0], rnn_running_avg_data[1]-rnn_running_avg_data[2], rnn_running_avg_data[1]+rnn_running_avg_data[2], color='#AFE9FF')

    
    #
    plt.legend(fontsize="15")
    # plt.title('Training Reward Comparison - RNN vs MARL')
    plt.ylabel('Reward', fontsize=15)
    plt.xlabel('Episode', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'training_reward_comparison_9.0_budget_{budget}.png')
    plt.show()




# # plot_learning_curve(episode, rnn_training_reward, 'training_plot.png', 'marl v1 training reward', 'running average reward')
# # print(len(marl_v2_training_reward))


# # cols = 2
# # rows = 4 // cols 
# # fig, ax = plt.subplots(rows, cols, figsize=(10,10))

