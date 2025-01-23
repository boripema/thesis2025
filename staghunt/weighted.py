import mesa_gym.gyms.grid.staghunt.env as w
import gymnasium as gym
import torch
from mesa_gym.trainers.DQN import DQN
import numpy as np
import pickle
import time
import os

env = w.MesaGoalEnv(render_mode=None, map=None)
path = os.path.dirname(os.path.abspath(__file__))

dqn_trained_models = {
    0: {
        "hunger": os.path.join(path, "models/HunterA_0_DQNlearning_hunger_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt"),
        "sustainability": os.path.join(path, "models/HunterA_0_DQNlearning_sustainability_1000_32_0.005_0.001_0.95_1.0_0.001_0.1.pt"),
        "social": os.path.join(path, "models/HunterA_0_DQNlearning_social_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt")
    },
    1: {
        "hunger": os.path.join(path, "models/HunterB_1_DQNlearning_hunger_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt"),
        "sustainability": os.path.join(path, "models/HunterB_1_DQNlearning_sustainability_1000_32_0.005_0.001_0.95_1.0_0.001_0.1.pt"),
        "social": os.path.join(path, "models/HunterB_1_DQNlearning_social_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt")
    }
}


# LOAD... dqn_models
dqn_models = {}
dqn_models = {0: {}, 1: {}}
for agent_id in dqn_trained_models.keys():
    dqn_models[agent_id] = {}
    for value_dim in dqn_trained_models[agent_id]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nb_actions = gym.spaces.flatdim(env.action_space[agent_id])
        nb_states = gym.spaces.flatdim(env.observation_space)
        
        model = DQN(nb_states, nb_actions).to(device)
        model.load_state_dict(torch.load(dqn_trained_models[agent_id][value_dim]))
        model.eval()
        dqn_models[agent_id][value_dim] = model


def combine_dqn_q_values(agent_id, state, dqn_models, weights):
    # Prepare the state tensor
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # Get Q-values from both value dimensions
    q_hunger = dqn_models[agent_id]["hunger"](state_tensor).detach().cpu().numpy().squeeze()
    q_sustainability = dqn_models[agent_id]["sustainability"](state_tensor).detach().cpu().numpy().squeeze()
    q_social = dqn_models[agent_id]["social"](state_tensor).detach().cpu().numpy().squeeze()

    # Combine Q-values using the weights
    combined_q_values = (
        weights["hunger"] * q_hunger +
        weights["sustainability"] * q_sustainability +
        weights["social"] * q_social
    )

    return combined_q_values

def value_alignment_filter(combined_q_values, ethical_q_values, threshold):
    filtered_q_values = combined_q_values.copy()

    for action in range(len(combined_q_values)):
        if ethical_q_values[action] < threshold:
            filtered_q_values[action] = -np.inf

    return filtered_q_values


# WEIGHTS
weights = {
    "hunger": 1.0,  
    "sustainability": 0.0,
    "social": 0.0
}

threshold = -np.inf  
num_episodes = 500

# PLAYING LOOP
def play_loop(weights, random=False):
    outcomes = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0

        model = "dqn"

        for _ in range(500):

            actions = {}
            for agent in env._get_agents():
                id = agent.unique_id
                percepts = obs[id]

                if model == "dqn":
                    if id in dqn_trained_models:
                        combined_q_values = combine_dqn_q_values(id, percepts, dqn_models, weights)
                        #print(f"combined:{combined_q_values}\n")
                        state_tensor = torch.tensor(percepts, dtype=torch.float32, device=device).unsqueeze(0)
                        #print(f"state_tensor:{state_tensor}\n")

                        hunger_q_values = dqn_models[id]["hunger"](state_tensor).detach().cpu().numpy().squeeze()
                        sustainability_q_values = dqn_models[id]["sustainability"](state_tensor).detach().cpu().numpy().squeeze()
                        social_q_values = dqn_models[id]["social"](state_tensor).detach().cpu().numpy().squeeze()
                        social_q_values = social_q_values - np.max(social_q_values)  
                        #print(f"hunger:{hunger_q_values}\n")
                        #print(f"sustainability:{sustainability_q_values}\n")
                        #print(f"social:{social_q_values}\n")

                        filtered_q_values = value_alignment_filter(combined_q_values, social_q_values, threshold)
                        #print(f"filtered:{filtered_q_values}\n")

                        # choose the highest social_q_value if all filtered values are -inf
                        if random == False:
                            if np.all(filtered_q_values == -np.inf):
                                actions[id] = int(np.argmax(social_q_values))
                            else:
                                    actions[id] = int(np.argmax(filtered_q_values))
                        else: 
                            actions[id] = env.action_space[id].sample()

                else:
                    actions[id] = env.action_space[id].sample()

            obs, reward, terminated, truncated, info = env.step(actions=actions)

            if terminated or truncated:
                break
            
        outcomes.append(info)

        env.close()
    return outcomes

outcomes = play_loop(weights, random=False)
random = play_loop(weights, random=True)

cumulative_hunted = []
cumulative_starved = []
cumulative_extinction = []
total_hunted = 0
total_starved = 0
total_extinction = 0

cumulative_hunted_random = []
cumulative_starved_random = []
cumulative_extinction_random = []
total_hunted_random = 0
total_starved_random = 0
total_extinction_random = 0

for episode in outcomes:
    total_hunted += sum(agent.get('hunted', 0) for agent in episode.values())
    total_starved += sum(agent.get('starved', 0) for agent in episode.values())
    total_extinction += sum(agent.get('extinction', 0) for agent in episode.values())
    _100hunted = total_hunted / 100
    cumulative_hunted.append(_100hunted)
    cumulative_starved.append(total_starved)
    cumulative_extinction.append(total_extinction)
    
print(f"total starved: {total_starved}, total extinction: {total_extinction}, total hunted: {total_hunted}")

for episode in random:
    total_hunted_random += sum(agent.get('hunted', 0) for agent in episode.values())
    total_starved_random += sum(agent.get('starved', 0) for agent in episode.values())
    total_extinction_random += sum(agent.get('extinction', 0) for agent in episode.values())
    _100hunted_random = total_hunted_random / 100
    cumulative_hunted_random.append(_100hunted_random)
    cumulative_starved_random.append(total_starved_random)
    cumulative_extinction_random.append(total_extinction_random)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(cumulative_hunted, label="Cumulative Hunted (per 100)", color="green")
plt.plot(cumulative_starved, label="Cumulative Starved", color="red")
plt.plot(cumulative_extinction, label="Cumulative Extinction", color="blue")

plt.plot(cumulative_hunted_random, label="Cumulative 100 Hunted (random actions)", color="green", linestyle="--")
plt.plot(cumulative_starved_random, label="Cumulative Starved (Random actions)", color="red", linestyle="--")
plt.plot(cumulative_extinction_random, label="Cumulative Extinction (Random acitons)", color="blue", linestyle="--")

plt.xlabel("Episode", fontsize=16) 
plt.ylabel("Cumulative Count", fontsize=16)  
plt.title(f"Voracity value dimension", fontsize=16)  # Title font size

plt.legend(fontsize=12)  

# Adjust tick label sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True)
plt.ylim(bottom=0, top=max(cumulative_starved_random + cumulative_extinction_random + cumulative_hunted_random) + 10)

plt.show()
