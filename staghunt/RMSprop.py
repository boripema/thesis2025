import torch.optim as optim
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


# Central finite difference gradient approximation
def fdga(weights, sigma, num_runs):
    baseline_loss = averaged_loss_function(weights, num_runs)[0]
    gradient = np.zeros_like(weights)

    for i in range(len(weights)):
        e_i = np.zeros_like(weights)
        e_i[i] = 1
        perturbed_loss = averaged_loss_function(weights + sigma * e_i, num_runs)[0]
        gradient[i] = (perturbed_loss - baseline_loss) / sigma
    return gradient


def normalize_weights(weights):
    weights = np.maximum(weights, 0)  # Ensure weights are non-negative
    total = np.sum(weights)
    if total == 0:
        print(weights)
        raise ValueError("All weights are zero or negative. Cannot normalize.")
    return weights / total

def clip_gradient(gradient, max_norm=5.0):
    norm = np.linalg.norm(gradient)
    if norm > max_norm:
        gradient = gradient * (max_norm / norm)
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
            q_social = q_social - np.max(q_social)

            combined_q_values = (
                weights[0] * q_hunger +
                weights[1] * q_sustainability +
                weights[2] * q_social
            )

            filtered_q_values = value_alignment_filter(combined_q_values, q_social, threshold)

            if np.all(filtered_q_values == -np.inf):
                actions[id] = int(np.argmax(combined_q_values))
            else:
                actions[id] = int(np.argmax(combined_q_values))
            

        # Step in the environment
        obs, reward, terminated, truncated, info = env.step(actions)

        if terminated or truncated:
            break

    for agent in info:
        for event in env.info[agent]:
            outcomes[event] += info[agent][event]

    return outcomes["starved"] + outcomes["extinction"] - outcomes["hunted"] / 100, outcomes["starved"], outcomes["extinction"], outcomes["hunted"] / 100



def averaged_loss_function(weights, num_runs):
    # Initialize cumulative statistics
    cumulative_stats = {"loss": 0, "starved": 0, "extinction": 0, "hunted": 0}

    for _ in range(num_runs):
        loss, starved, extinction, hunted = loss_function(weights)
        
        # Accumulate the counts
        cumulative_stats["loss"] += loss
        cumulative_stats["starved"] += starved
        cumulative_stats["extinction"] += extinction
        cumulative_stats["hunted"] += hunted

    # Average the loss
    # cumulative_stats["loss"] /= num_runs

    return cumulative_stats["loss"], cumulative_stats["starved"], cumulative_stats["extinction"], cumulative_stats["hunted"]

weights = np.array([1/3, 1/3, 1/3])  # Initial weights for the Q-value combination
# RMSprop optimizer with initial learning rate
initial_lr = 0.01  # Set an appropriate learning rate for RMSprop
optimizer = optim.RMSprop([torch.tensor(weights, requires_grad=True)], lr=initial_lr)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

loss_history = []
weight_history = {"weight 1": [], "weight 2": [], "weight 3": []}
cumulative_stats = {"starved": [], "extinction": [], "hunted": []}


num_iterations = 100
num_runs= 10
sigma = 0.01

# Training loop
loss, starved, extinction, hunted = averaged_loss_function(weights, num_runs)
print(f"Initial loss: {loss}")
loss_history.append(loss)
for iteration in range(num_iterations):
    
    # Compute gradient using FDGA
    gradient = fdga(weights, sigma, num_runs)
    gradient = clip_gradient(gradient)

    # Convert the gradient to a tensor with matching dtype
    weights_tensor = optimizer.param_groups[0]['params'][0].data
    gradient_tensor = torch.tensor(gradient, dtype=weights_tensor.dtype, requires_grad=False)

    # Assign the gradient to the optimizer parameter
    optimizer.param_groups[0]['params'][0].grad = gradient_tensor

    # Perform optimization step
    optimizer.step()

    # Normalize updated weights
    weights_tensor = optimizer.param_groups[0]['params'][0].data
    weights = weights_tensor.numpy()
    weights[:3] = normalize_weights(weights[:3])

    
    # Save weight history
    for i in range(3):
        weight_history[f"weight {i+1}"].append(weights[i])

    loss, starved, extinction, hunted = averaged_loss_function(weights, num_runs)

    # Adjust the learning rate using the scheduler
    scheduler.step(loss)
    current_lr = optimizer.param_groups[0]['lr']

    # Record loss and cumulative statistics
    loss_history.append(loss)

    for key in cumulative_stats:
        cumulative_stats[key].append(locals()[key])

    print(f"Iteration {iteration}: Weights = {weights}, Loss = {loss}, LR = {current_lr}    Starved: {starved}, Extinction: {extinction}, Hunted: {hunted}")



# Training Loss Plot
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label="Loss", alpha=0.5)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()

# Weight Adjustment Plot
plt.figure(figsize=(10, 4))
plt.plot(weight_history["weight 1"], label="weight 1", color="black")
plt.plot(weight_history["weight 2"], label="weight 2", color="orange")
plt.plot(weight_history["weight 3"], label="weight 3", color="blue")
plt.xlabel("Gradient descent step")
plt.ylabel("Weights")
plt.title("Weight adjustment over training")
plt.legend()
plt.grid(True)
plt.ylim(bottom=0, top=1)
plt.show()

# Cumulative Statistics Plot
plt.figure(figsize=(10, 4))
plt.plot(cumulative_stats["hunted"], label="Cumulative hunted (per 100)", color="green")
plt.plot(cumulative_stats["starved"], label="Cumulative starved", color="red")
plt.plot(cumulative_stats["extinction"], label="Cumulative extinction", color="blue")
plt.xlabel("Gradient descent step")
plt.ylabel("Number of events")
plt.title(f"Cumulative agents starved, extinctions and stags hunted (per 100) in {num_runs} episodes")
plt.legend()
plt.grid(True)
plt.show()
