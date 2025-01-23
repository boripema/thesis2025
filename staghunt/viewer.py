import torch
import numpy as np
import os
import mesa_gym.gyms.grid.staghunt.env as w
from mesa_gym.trainers.DQN import DQN

# Load environment
env = w.MesaGoalEnv(render_mode=None, map=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path setup
path = os.path.dirname(os.path.abspath(__file__))
dqn_trained_models = {
    0: {
        "hunger": os.path.join(path, "models/HunterA_0_DQNlearning_hunger_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt"),
        "sustainability": os.path.join(path, "models/HunterA_0_DQNlearning_sustainability_1000_32_0.005_0.001_0.95_1.0_0.001_0.1.pt"),
        "social": os.path.join(path, "models/HunterA_0_DQNlearning_social_1000_32_0.005_0.001_0.95_1.0_0.002_0.1.pt")
    },
}

# Load DQN model
dqn_models = {}
for agent_id in dqn_trained_models.keys():
    dqn_models[agent_id] = {}
    for value_dim in dqn_trained_models[agent_id]:
        nb_actions = env.action_space[agent_id].n
        nb_states = env.observation_space.shape[0]
        model = DQN(nb_states, nb_actions).to(device)
        model.load_state_dict(torch.load(dqn_trained_models[agent_id][value_dim], map_location=device))
        model.eval()
        dqn_models[agent_id][value_dim] = model

# Generate random states and get Q-values
def print_random_states(dqn_model, num_states=20):
    """
    Prints Q-values for random states.
    :param dqn_model: The trained DQN model
    :param num_states: Number of random states to print
    """
    for i in range(num_states):
        # Generate a random state
        random_state = np.random.rand(env.observation_space.shape[0])  # Random state within [0, 1]
        state_tensor = torch.tensor(random_state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Get Q-values
        q_values = dqn_model(state_tensor).detach().cpu().numpy().squeeze()
        
        # Print the state and corresponding Q-values
        print(f"State {i + 1}: {random_state}")
        print(f"Q-values: {q_values}")
        print("-" * 30)

# Print Q-values for a specific model (adjust agent ID and value dimension as needed)
print("Random Q-values from 'hunger' model of Agent 0:")
print("HUNGER")
print_random_states(dqn_models[0]["hunger"], num_states=10)
print("SUSTAINABILITY")
print_random_states(dqn_models[0]["sustainability"], num_states=10)
print("SOCIAL")
print_random_states(dqn_models[0]["social"], num_states=10)
