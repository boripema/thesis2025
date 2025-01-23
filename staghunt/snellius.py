import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import numpy as np
import pickle
import time
import os
import sys

sys.path.append('/gpfs/home6/bipema/mesa-gym')
path = '/gpfs/home6/bipema/mesa-gym/mesa_gym/gyms/grid/staghunt'

import mesa_gym.gyms.grid.staghunt.env as w
from mesa_gym.trainers.DQN import DQN
from mesa.time import RandomActivation


env = w.MesaGoalEnv(render_mode=None, map=None)
#path = os.path.dirname(os.path.abspath(__file__))


dqn_trained_models = {
    0: {
        "hunger": os.path.join(path, "models/HunterA_0_DQNlearning_hunger_10000_32_0.005_0.001_0.95_1.0_0.0002_0.1.pt"),
        "sustainability": os.path.join(path, "models/HunterA_0_DQNlearning_sustainability_10000_32_0.005_0.001_0.95_1.0_0.0002_0.1.pt"),
        "social": os.path.join(path, "models/HunterA_0_DQNlearning_social_10000_32_0.005_0.001_0.95_1.0_0.0002_0.1.pt")
    },
    1: {
        "hunger": os.path.join(path, "models/HunterB_1_DQNlearning_hunger_10000_32_0.005_0.001_0.95_1.0_0.0002_0.1.pt"),
        "sustainability": os.path.join(path, "models/HunterB_1_DQNlearning_sustainability_10000_32_0.005_0.001_0.95_1.0_0.0002_0.1.pt"),
        "social": os.path.join(path, "models/HunterB_1_DQNlearning_social_10000_32_0.005_0.001_0.95_1.0_0.0002_0.1.pt")
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
    "hunger": 0.06018504,  
    "sustainability": 0.55546498,
    "social": 0.38434999
}

thresholds = [-7.5, -5, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 5, 7.5]
num_episodes = 5

cumulative_hunted = []
cumulative_starved = []
cumulative_extinction = []

def run(weights, threshold):
    outcomes = []  # Reset outcomes for each threshold

    # PLAYING LOOP
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0

        model = "dqn"

        for _ in range(500):

            actions = {}
            for agent in env._get_agents():
                id = agent.unique_id
                percepts = obs[id]
                
                """
                if model == "q-learning":
                    if id in q_trained_models:
                        if tuple(percepts) not in combined_q_tables[id]:
                            raise RuntimeError("Warning: state not present in the Q-table, have you loaded the correct one?")
                        
                        combined_q_values = combined_q_tables[id][tuple(percepts)]
                        sustainability_q_values = q_tables[id]["sustainability"][tuple(percepts)]
                        filtered_q_values = value_alignment_filter(combined_q_values, sustainability_q_values, threshold)

                        action = int(np.argmax(filtered_q_values))
                """

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
                        if np.all(filtered_q_values == -np.inf):
                            actions[id] = int(np.argmax(social_q_values))
                        else:
                            actions[id] = int(np.argmax(filtered_q_values))

                else:
                    actions[id] = env.action_space[id].sample()

            obs, reward, terminated, truncated, info = env.step(actions=actions)

            if terminated or truncated:
                break
            
        outcomes.append(info)

        env.close()

    return outcomes

start = time.time()
outcomes = [run(weights, threshold) for threshold in thresholds] 
end = time.time()

no_filter_outcomes = run(weights, -np.inf)

cumulative_hunted = []
cumulative_starved = []
cumulative_extinction = []

# Loop through outcomes for each threshold
for threshold_outcomes in outcomes:
    total_hunted = 0
    total_starved = 0
    total_extinction = 0

    # Calculate cumulative metrics for each threshold
    for episode_outcome in threshold_outcomes:
        total_hunted += sum(agent.get('hunted', 0) for agent in episode_outcome.values())
        total_starved += sum(agent.get('starved', 0) for agent in episode_outcome.values())
        total_extinction += sum(agent.get('extinction', 0) for agent in episode_outcome.values())
    print(f"total starved: {total_starved}, total extinction: {total_extinction}, total hunted: {total_hunted}")

    # Normalize or aggregate as needed (e.g., divide hunted by 100)
    total_hunted /= 100  # Normalize per 100 episodes
    cumulative_hunted.append(total_hunted)
    cumulative_starved.append(total_starved)  # Average per episode
    cumulative_extinction.append(total_extinction)  # Average per episode


inf_avg_hunted = 0
inf_avg_starved = 0
inf_avg_extinction = 0

for episode in no_filter_outcomes:
    inf_avg_hunted += sum(agent.get('hunted', 0) for agent in episode.values())
    inf_avg_starved += sum(agent.get('starved', 0) for agent in episode.values())
    inf_avg_extinction += sum(agent.get('extinction', 0) for agent in episode.values())
print(f"total starved: {inf_avg_starved}, total extinction: {inf_avg_extinction}, total hunted: {inf_avg_hunted}, threshold: -inf")
inf_avg_hunted /= 100  # Normalize per 100 episodes

# Plotting
plt.figure(figsize=(30, 5))

# Plot cumulative values
plt.plot(thresholds, cumulative_hunted, label="Cumulative Hunted (per 100)", color="green", marker="o")
plt.plot(thresholds, cumulative_starved, label="Cumulative Starved", color="red", marker="o")
plt.plot(thresholds, cumulative_extinction, label="Cumulative Extinction", color="blue", marker="o")

# Add horizontal striped lines for -np.inf reference
plt.axhline(y=inf_avg_hunted, color="green", linestyle="--", linewidth=1, label="Hunted Reference (no filter)")
plt.axhline(y=inf_avg_starved, color="red", linestyle="--", linewidth=1, label="Starved Reference (no filter)")
plt.axhline(y=inf_avg_extinction, color="blue", linestyle="--", linewidth=1, label="Extinction Reference (no filter)")

# Set x-ticks explicitly
plt.xticks(ticks=thresholds, labels=[f"{t}" for t in thresholds], rotation=45, fontsize=12)  # Rotate for clarity
plt.yticks(fontsize=12)

# Labels and title
plt.xlabel("Threshold", fontsize=16)
plt.ylabel("Cumulative Count", fontsize=16)
plt.title(f"Cumulative statistics with different thresholds averaged over {num_episodes} episodes", fontsize=16)

# Legend
plt.legend(fontsize=16)

# Grid and scaling
plt.grid(True)
plt.ylim(bottom=0, top=max(cumulative_starved + cumulative_extinction + cumulative_hunted) + 10)  # Adjust top dynamically

# Display the plot
plt.show()
