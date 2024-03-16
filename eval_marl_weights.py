import pandas as pd 

df = pd.read_csv('project/code/model/marl_v1/test_2/ep_2000_10000/budget_6000_6000/num_agent_3_3/marl_v1_weights/starting_city_0/model_weight.csv')

q_table = df['q_table']
q_table = q_table.to_dict()

r_table = df['r_table']
r_table = r_table.to_dict()

max_ = 0
max_pos = None 
# for i in range(len(r_table)):

#     for key, val in r_table[i].item():
#         for r_val in val:
#             continue

print(r_table[len(r_table)-1])

# print(f"max_q: {}\nmax_r:{df['r_table'].max()}")