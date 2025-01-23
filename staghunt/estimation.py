import mesa_gym.gyms.grid.staghunt.env as w
import gymnasium as gym
import torch
import numpy as np
from mesa_gym.trainers.DQN import DQN
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

env = w.MesaGoalEnv(render_mode=None , map=None)
path = os.path.dirname(os.path.abspath(__file__))

q_trained_models = {
    0: {
        "hunger": os.path.join(path, "models/HunterA_0_qlearning_hunger_10000_0.01_0.95_1.0_0.0002_0.1.pickle"),
        "sustainability": os.path.join(path, "models/HunterA_0_qlearning_sustainability_10000_0.01_0.95_1.0_0.0002_0.1.pickle"),
        "social": os.path.join(path, "models/HunterA_0_qlearning_social_10000_0.01_0.95_1.0_0.0002_0.1.pickle")
    },
    1: {
        "hunger": os.path.join(path, "models/HunterB_1_qlearning_hunger_10000_0.01_0.95_1.0_0.0002_0.1.pickle"),
        "sustainability": os.path.join(path, "models/HunterB_1_qlearning_sustainability_10000_0.01_0.95_1.0_0.0002_0.1.pickle"),
        "social": os.path.join(path, "models/HunterB_1_qlearning_social_10000_0.01_0.95_1.0_0.0002_0.1.pickle")
    }
}

dqn_trained_models = {
    0: {
        "hunger": os.path.join(path, "models/HunterA_0_DQNlearning_hunger_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt"),
        "sustainability": os.path.join(path, "models/HunterA_0_DQNlearning_sustainability_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt"),
        "social": os.path.join(path, "models/HunterA_0_DQNlearning_social_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt")
    },
    1: {
        "hunger": os.path.join(path, "models/HunterB_1_DQNlearning_hunger_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt"),
        "sustainability": os.path.join(path, "models/HunterB_1_DQNlearning_sustainability_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt"),
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


weights = np.array([1/3, 1/3, 1/3])  # Initial weights for the Q-value combination

# FDGA Hyperparameters
sigma = 0.01  # Perturbation step size
alpha = 0.01   # Learning rate
num_iterations = 1


# Central finite difference gradient approximation
def ffd(weights, sigma, num_runs=500):
    baseline_loss = averaged_loss_function(weights, num_runs)[0]
    gradient = np.zeros_like(weights)

    for i in range(len(weights)):
        e_i = np.zeros_like(weights)
        e_i[i] = 1
        perturbed_loss = averaged_loss_function(weights + sigma * e_i, num_runs)[0]
        gradient[i] = (perturbed_loss - baseline_loss) / sigma
    return gradient


def value_alignment_filter(combined_q_values, ethical_q_values, threshold):
    filtered_q_values = combined_q_values.copy()

    for action in range(len(combined_q_values)):
        if ethical_q_values[action] < threshold:
            filtered_q_values[action] = -np.inf

    return filtered_q_values

threshold = -np.inf


def loss_function(weights):
    obs, info = env.reset()
    outcomes = {"hunted": 0, "starved": 0, "extinction": 0}

    for _ in range(500):  
        actions = {}
        for agent in env._get_agents():
            id = agent.unique_id
            percepts = obs[id]

            # Compute combined Q-values based on weights
            state_tensor = torch.tensor(percepts, dtype=torch.float32, device=device).unsqueeze(0)
            q_hunger = dqn_models[id]["hunger"](state_tensor).detach().cpu().numpy().squeeze()
            q_sustainability = dqn_models[id]["sustainability"](state_tensor).detach().cpu().numpy().squeeze()
            q_social = dqn_models[id]["social"](state_tensor).detach().cpu().numpy().squeeze()
            q_social = q_social - max(q_social) 

            combined_q_values = (
                weights[0] * q_hunger +
                weights[1] * q_sustainability +
                weights[2] * q_social
            )

            filtered_q_values = value_alignment_filter(combined_q_values, q_social, threshold)

            if np.all(filtered_q_values == -np.inf):
                actions[id] = int(np.argmax(q_social))
            else:
                actions[id] = int(np.argmax(filtered_q_values))
            

        # Step in the environment
        obs, reward, terminated, truncated, info = env.step(actions)

        # Update outcomes based on the episode
        if terminated or truncated:
            break

    for agent in info:
        for event in env.info[agent]:
            outcomes[event] += info[agent][event]

    # Compute the loss based on outcomes
    return outcomes["starved"] + outcomes["extinction"] - (outcomes["hunted"] / 150), outcomes["starved"], outcomes["extinction"], outcomes["hunted"] / 100



#def averaged_loss_function(weights, num_runs):
        #return np.mean([loss_function(weights) for _ in range(num_runs)])

def averaged_loss_function(weights, num_runs):
    all_stats = np.array([loss_function(weights) for _ in range(num_runs)])
    all_stats[0] = np.mean(all_stats[:, 0])
    return all_stats


def normalize_weights(weights):
    weights = np.maximum(weights, 0)  # Ensure weights are non-negative
    total = np.sum(weights)
    if total == 0:
        print(weights)
        raise ValueError("All weights are zero or negative. Cannot normalize.")
    return weights / total

def clip_gradient(gradient, max_norm=1.0):
    norm = np.linalg.norm(gradient)
    if norm > max_norm:
        gradient = gradient * (max_norm / norm)
    return gradient


loss_history = []
weight_history = {"weight 1": [], "weight 2": [], "weight 3": []}
cumulative_stats = {"starved": [], "extinction": [], "hunted": []}

# RL training loop with FDGA-based optimization
for iteration in range(num_iterations):
    # Compute gradient using FDGA
    gradient = ffd(weights, sigma)
    gradient = clip_gradient(gradient)
    print(f"Gradient: {gradient}")
    
    # Update weights
    weights -= alpha * gradient
    weights[:3] = normalize_weights(weights[:3])

    for i in range(3):
        weight_history[f"weight {i+1}"].append(weights[i])

    # Evaluate the updated weights
    loss, starved, extinction, hunted= averaged_loss_function(weights, num_runs=500)

    loss_history.append(loss)
    
    for key in cumulative_stats:
        cumulative_stats[key].append(locals()[key])

    print(f"Iteration {iteration}: Weights = {weights}, Loss = {loss}")


# Training Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label="Loss", alpha=0.5)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()

# Weight Adjustment Plot
plt.figure(figsize=(10, 6))
plt.plot(weight_history["weight 1"], label="weight 1", color="black")
plt.plot(weight_history["weight 2"], label="weight 2", color="orange")
plt.plot(weight_history["weight 3"], label="weight 3", color="blue")
plt.xlabel("Gradient descent step")
plt.ylabel("Weights")
plt.title("Weight adjustment over training")
plt.legend()
plt.grid(True)
plt.show()

# Cumulative Statistics Plot
plt.figure(figsize=(10, 6))
plt.plot(cumulative_stats["hunted"], label="Cumulative hunted / 150", color="green")
plt.plot(cumulative_stats["starved"], label="Cumulative starved", color="red")
plt.plot(cumulative_stats["extinction"], label="Cumulative extinction", color="blue")
plt.xlabel("Gradient descent step")
plt.ylabel("Number of events")
plt.title("Average cumulative statistics for 200 episodes over training")
plt.legend()
plt.grid(True)
plt.show()

