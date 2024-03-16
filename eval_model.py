import torch 
from dqn_network import DQN, DQN_RNN
from sensor_network import Network
from itertools import count
import torch.nn.functional as F
from agent import Agent

def load_model(hidden_1, hidden_2, name):
    path = f'project/code/model/city_10/test_3/model_size_{hidden_1}_{hidden_2}/batch_32_lr_0.001_memoryBuffer_200000/starting_city_0/{name}.tar'
    checkpoint = torch.load(path)
    # policy_net = DQN(10, 10, hidden_1, hidden_2)
    policy_net = DQN_RNN(num_cities, hidden_1, hidden_2)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    return policy_net

hidden_size_1 = [64, 128]
hidden_size_2 = [128, 256]
model_names = ['rnn_dqn']
pred = {}

# sample_network = Network(can_revisit=False).build_city_sample(start_city=0, unit='mile')
start_city=0
for hidden_1 in hidden_size_1:
    for hidden_2 in hidden_size_2:
        for name in model_names:
            print("====================================================")
            print("====================================================")
            print("====================================================")
            path = [start_city]
            network = Network(can_revisit=False).build_city_sample(start_city=start_city, unit='mile')
            num_cities = network.num_node
            model = load_model(hidden_1, hidden_2, name)
            agent = Agent(None, num_cities, model, model, None, lr=0.0001, gamma=None, agent_name=name, budget=6000, is_training=False)
            for timestep in count():
                state = network.current_nodes(one_hot_encoding=False)
                state = F.one_hot(torch.tensor([state]), num_classes=num_cities).to(torch.float32)

                state_location = network.state_location()
                state_location = torch.tensor(state_location, dtype=torch.float32).unsqueeze(0)

                feasible_mask = network.get_feasible_mask(agent.current_budget())
                feasible_mask = torch.tensor(feasible_mask, dtype=torch.float32).view(1, 1, -1)
                # print(f"feasible_mask: {feasible_mask}")
                
                action = agent.select_action(state, state_location, feasible_mask)
                path.append(action)

                next_state_representation, reward, is_done, next_state_location, cost = network.visit(action)
                agent.collect_prize_n_adjust_budget(reward, cost)

                if is_done:
                    if action != start_city:
                        cost_to_starting = network.min_cost_graph[network.current_nodes()]['min_cost']
                        agent.collect_prize_n_adjust_budget(0, cost_to_starting)
                        path.append(start_city) #travel back to starting city
                    break 
        print(f"model_{hidden_1}_{hidden_2} at city {start_city}: {agent.collected_prizes()}\nPred path: {path}")


'''
hidden_size_1 = 64
# [64, 128]
hidden_size_2 = 128
path  = 'project/code/model/city_10/test_1/model_size_64_128/batch_64_lr_0.001_memoryBuffer_200000/starting_city_0/simple_dqn.tar'
# 'project/code/model/simple_dqn.tar'

checkpoint = torch.load(path)

policy_net = DQN(10, 10, hidden_size_1, hidden_size_2)
print(f"before: {policy_net.state_dict()}")

policy_net.load_state_dict(checkpoint['model_state_dict'])
print(f"after: {policy_net.state_dict()}")

# print(policy_net.eval())
'''