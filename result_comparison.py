import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


rnn_training_df = pd.read_csv('project/code/model/rnn/city_10/test_4/training_data.csv')
rnn_training_df_details = rnn_training_df.groupby(['starting_city',  'total_episode', 'total_budget'])['max_reward'].max().reset_index().sort_values(by=['starting_city', 'total_episode', 'total_budget'])
rnn_training_df_details = rnn_training_df_details.rename(columns={'max_reward':'rnn_training_max_reward'})
rnn_training_df_details = rnn_training_df_details.set_index(['starting_city',  'total_episode', 'total_budget'])

rnn_prediction_df = pd.read_csv('project/code/model/rnn/city_10/test_4/prediction_data.csv')
rnn_prediction_df_details = rnn_prediction_df.groupby(['starting_city',  'total_episode', 'total_budget'])['reward'].max().reset_index().sort_values(by=['starting_city', 'total_episode', 'total_budget'])
rnn_prediction_df_details = rnn_prediction_df_details.rename(columns={'reward':'rnn_prediction_reward'})
rnn_prediction_df_details = rnn_prediction_df_details.set_index(['starting_city',  'total_episode', 'total_budget'])

''' MARL V1 
marl_v1_training_df = pd.read_csv('project/code/model/marl_v1/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/training/marl_10_city_2.csv')
marl_v1_training_df_details = marl_v1_training_df.groupby(['starting_city', 'num_agent', 'total_episode', 'total_budget'])['max_reward'].max().reset_index().sort_values(by=['starting_city', 'num_agent', 'total_episode', 'total_budget'])

marl_v1_training_df_details_3_agents = marl_v1_training_df_details[marl_v1_training_df_details['num_agent'] == 3]
marl_v1_training_df_details_3_agents = marl_v1_training_df_details_3_agents.drop(columns=['num_agent'])
marl_v1_training_df_details_3_agents = marl_v1_training_df_details_3_agents.rename(columns={'max_reward':'marl_v1_3_agents_training_max_reward'})
marl_v1_training_df_details_3_agents = marl_v1_training_df_details_3_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v1_training_df_details_5_agents = marl_v1_training_df_details[marl_v1_training_df_details['num_agent'] == 5]
marl_v1_training_df_details_5_agents = marl_v1_training_df_details_5_agents.drop(columns=['num_agent'])
marl_v1_training_df_details_5_agents = marl_v1_training_df_details_5_agents.rename(columns={'max_reward':'marl_v1_5_agents_training_max_reward'})
marl_v1_training_df_details_5_agents = marl_v1_training_df_details_5_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v1_training_df_details_10_agents = marl_v1_training_df_details[marl_v1_training_df_details['num_agent'] == 10]
marl_v1_training_df_details_10_agents = marl_v1_training_df_details_10_agents.drop(columns=['num_agent'])
marl_v1_training_df_details_10_agents = marl_v1_training_df_details_10_agents.rename(columns={'max_reward':'marl_v1_10_agents_training_max_reward'})
marl_v1_training_df_details_10_agents = marl_v1_training_df_details_10_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v1_prediction_df = pd.read_csv('project/code/model/marl_v1/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/execution/marl_10_city_2.csv')
marl_v1_prediction_df_details = marl_v1_prediction_df.groupby(['starting_city', 'num_agent', 'total_episode', 'total_budget'])['reward'].max().reset_index().sort_values(by=['starting_city', 'num_agent', 'total_episode', 'total_budget'])

marl_v1_prediction_df_details_3_agents = marl_v1_prediction_df_details[marl_v1_prediction_df_details['num_agent'] == 3]
marl_v1_prediction_df_details_3_agents = marl_v1_prediction_df_details_3_agents.drop(columns=['num_agent'])
marl_v1_prediction_df_details_3_agents = marl_v1_prediction_df_details_3_agents.rename(columns={'reward':'marl_v1_3_agents_pre_reward'})
marl_v1_prediction_df_details_3_agents = marl_v1_prediction_df_details_3_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v1_prediction_df_details_5_agents = marl_v1_prediction_df_details[marl_v1_prediction_df_details['num_agent'] == 5]
marl_v1_prediction_df_details_5_agents = marl_v1_prediction_df_details_5_agents.drop(columns=['num_agent'])
marl_v1_prediction_df_details_5_agents = marl_v1_prediction_df_details_5_agents.rename(columns={'reward':'marl_v1_5_agents_pre_reward'})
marl_v1_prediction_df_details_5_agents = marl_v1_prediction_df_details_5_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v1_prediction_df_details_10_agents = marl_v1_prediction_df_details[marl_v1_prediction_df_details['num_agent'] == 10]
marl_v1_prediction_df_details_10_agents = marl_v1_prediction_df_details_10_agents.drop(columns=['num_agent'])
marl_v1_prediction_df_details_10_agents = marl_v1_prediction_df_details_10_agents.rename(columns={'reward':'marl_v1_10_agents_pre_reward'})
marl_v1_prediction_df_details_10_agents = marl_v1_prediction_df_details_10_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v1_df = pd.concat([marl_v1_training_df_details_3_agents, marl_v1_training_df_details_5_agents, marl_v1_training_df_details_10_agents
                        ,marl_v1_prediction_df_details_3_agents,  marl_v1_prediction_df_details_5_agents, marl_v1_prediction_df_details_10_agents], axis=1, join='inner').reset_index()


marl_v1_df_city_0_budget_6000 = marl_v1_df[(marl_v1_df['starting_city'] == 0) & (marl_v1_df['total_budget'] == 6000)]

marl_v1_df.to_csv('project/code/model/marl_v1/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/training_vs_prediction.csv', index=False)

marl_v1_df_city_0_budget_6000 = marl_v1_df[(marl_v1_df['starting_city'] == 0) & (marl_v1_df['total_budget'] == 6000)]

marl_v1_training_3_agents = marl_v1_df_city_0_budget_6000['marl_v1_3_agents_training_max_reward']
marl_v1_training_5_agents = marl_v1_df_city_0_budget_6000['marl_v1_5_agents_training_max_reward']
marl_v1_training_10_agents = marl_v1_df_city_0_budget_6000['marl_v1_10_agents_training_max_reward']

marl_v1_prediction_3_agents = marl_v1_df_city_0_budget_6000['marl_v1_3_agents_pre_reward']
marl_v1_prediction_5_agents = marl_v1_df_city_0_budget_6000['marl_v1_5_agents_pre_reward']
marl_v1_prediction_10_agents = marl_v1_df_city_0_budget_6000['marl_v1_10_agents_pre_reward']
'''

''' MARL V2 
marl_v2_training_df = pd.read_csv('project/code/model/marl_v2/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/training/marl_10_city_2.csv')
marl_v2_training_df_details = marl_v2_training_df.groupby(['starting_city', 'num_agent', 'total_episode', 'total_budget'])['max_reward'].max().reset_index().sort_values(by=['starting_city', 'num_agent', 'total_episode', 'total_budget'])

marl_v2_training_df_details_3_agents = marl_v2_training_df_details[marl_v2_training_df_details['num_agent'] == 3]
marl_v2_training_df_details_3_agents = marl_v2_training_df_details_3_agents.drop(columns=['num_agent'])
marl_v2_training_df_details_3_agents = marl_v2_training_df_details_3_agents.rename(columns={'max_reward':'marl_v2_3_agents_training_max_reward'})
marl_v2_training_df_details_3_agents = marl_v2_training_df_details_3_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v2_training_df_details_5_agents = marl_v2_training_df_details[marl_v2_training_df_details['num_agent'] == 5]
marl_v2_training_df_details_5_agents = marl_v2_training_df_details_5_agents.drop(columns=['num_agent'])
marl_v2_training_df_details_5_agents = marl_v2_training_df_details_5_agents.rename(columns={'max_reward':'marl_v2_5_agents_training_max_reward'})
marl_v2_training_df_details_5_agents = marl_v2_training_df_details_5_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v2_training_df_details_10_agents = marl_v2_training_df_details[marl_v2_training_df_details['num_agent'] == 10]
marl_v2_training_df_details_10_agents = marl_v2_training_df_details_10_agents.drop(columns=['num_agent'])
marl_v2_training_df_details_10_agents = marl_v2_training_df_details_10_agents.rename(columns={'max_reward':'marl_v2_10_agents_training_max_reward'})
marl_v2_training_df_details_10_agents = marl_v2_training_df_details_10_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v2_prediction_df = pd.read_csv('project/code/model/marl_v2/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/execution/marl_10_city_2.csv')
marl_v2_prediction_df_details = marl_v2_prediction_df.groupby(['starting_city', 'num_agent', 'total_episode', 'total_budget'])['reward'].max().reset_index().sort_values(by=['starting_city', 'num_agent', 'total_episode', 'total_budget'])

marl_v2_prediction_df_details_3_agents = marl_v2_prediction_df_details[marl_v2_prediction_df_details['num_agent'] == 3]
marl_v2_prediction_df_details_3_agents = marl_v2_prediction_df_details_3_agents.drop(columns=['num_agent'])
marl_v2_prediction_df_details_3_agents = marl_v2_prediction_df_details_3_agents.rename(columns={'reward':'marl_v2_3_agents_pre_reward'})
marl_v2_prediction_df_details_3_agents = marl_v2_prediction_df_details_3_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v2_prediction_df_details_5_agents = marl_v2_prediction_df_details[marl_v2_prediction_df_details['num_agent'] == 5]
marl_v2_prediction_df_details_5_agents = marl_v2_prediction_df_details_5_agents.drop(columns=['num_agent'])
marl_v2_prediction_df_details_5_agents = marl_v2_prediction_df_details_5_agents.rename(columns={'reward':'marl_v2_5_agents_pre_reward'})
marl_v2_prediction_df_details_5_agents = marl_v2_prediction_df_details_5_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v2_prediction_df_details_10_agents = marl_v2_prediction_df_details[marl_v2_prediction_df_details['num_agent'] == 10]
marl_v2_prediction_df_details_10_agents = marl_v2_prediction_df_details_10_agents.drop(columns=['num_agent'])
marl_v2_prediction_df_details_10_agents = marl_v2_prediction_df_details_10_agents.rename(columns={'reward':'marl_v2_10_agents_pre_reward'})
marl_v2_prediction_df_details_10_agents = marl_v2_prediction_df_details_10_agents.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_v2_df = pd.concat([marl_v2_training_df_details_3_agents, marl_v2_training_df_details_5_agents, marl_v2_training_df_details_10_agents
                        ,marl_v2_prediction_df_details_3_agents,  marl_v2_prediction_df_details_5_agents, marl_v2_prediction_df_details_10_agents], axis=1, join='inner').reset_index()


marl_v2_df_city_0_budget_6000 = marl_v2_df[(marl_v2_df['starting_city'] == 0) & (marl_v2_df['total_budget'] == 6000)]

marl_v2_df.to_csv('project/code/model/marl_v2/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/training_vs_prediction.csv', index=False)

marl_v2_df_city_0_budget_6000 = marl_v2_df[(marl_v2_df['starting_city'] == 0) & (marl_v2_df['total_budget'] == 6000)]

marl_v2_training_3_agents = marl_v2_df_city_0_budget_6000['marl_v2_3_agents_training_max_reward']
marl_v2_training_5_agents = marl_v2_df_city_0_budget_6000['marl_v2_5_agents_training_max_reward']
marl_v2_training_10_agents = marl_v2_df_city_0_budget_6000['marl_v2_10_agents_training_max_reward']

marl_v2_prediction_3_agents = marl_v2_df_city_0_budget_6000['marl_v2_3_agents_pre_reward']
marl_v2_prediction_5_agents = marl_v2_df_city_0_budget_6000['marl_v2_5_agents_pre_reward']
marl_v2_prediction_10_agents = marl_v2_df_city_0_budget_6000['marl_v2_10_agents_pre_reward']
'''



# print(marl_v2_df_city_0_budget_6000)



rnn_df = pd.concat([rnn_training_df_details, rnn_prediction_df_details], axis=1, join='inner').reset_index()
rnn_df.to_csv('project/code/model/rnn/city_10/test_4/training_vs_prediction.csv', index=False)

rnn_df_city_0_budget_6000 = rnn_df[(rnn_df['starting_city']==0) & (rnn_df['total_budget'] == 6000)]




episodes = rnn_df_city_0_budget_6000['total_episode']
rnn_training = rnn_df_city_0_budget_6000['rnn_training_max_reward']
rnn_pred = rnn_df_city_0_budget_6000['rnn_prediction_reward']



# # print(marl_v2_training_5_agents)
opt_solution = [479,481,495,505,514,529,536,542,551,551]

barWidth = 0.20
br1 = np.arange(len(episodes)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
# # br4 = [x + barWidth for x in br3] 
# # br5 = [x + barWidth for x in br4] 
# # br6 = [x + barWidth for x in br5] 

plt.bar(br1, opt_solution[0], color='g', width=barWidth, edgecolor ='grey', label ='optimal_reward')
# plt.bar(br2, rnn_training, color='b', width=barWidth, edgecolor ='grey', label ='training_max_reward')
# plt.bar(br3, rnn_pred, color='r', width=barWidth, edgecolor ='grey', label ='prediction_reward')
plt.bar(br2, marl_v2_training_10_agents, color='b', width=barWidth, edgecolor ='grey', label ='marl_v2_10_agents_training_max_reward')
plt.bar(br3, marl_v2_prediction_10_agents, color='r', width=barWidth, edgecolor ='grey', label ='marl_v2_10_agents_prediction_reward')

plt.xlabel('episode')
plt.ylabel('reward')
# plt.title('Starting City: 0 - RNN performance comparison')
plt.title('Starting City: 0 - MARL_v2 performance comparison (10 agents)')
plt.xticks([r + barWidth for r in range(len(episodes))], 
        episodes)
plt.legend()
# plt.savefig("marl_v2_10_agents_comparison.png") 
plt.show()


