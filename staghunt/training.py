import os
import sys
import pickle

#sys.path.append('/gpfs/home6/bipema/mesa-gym')
#path = '/gpfs/home6/bipema/mesa-gym/mesa_gym/gyms/grid/staghunt'

path = os.path.dirname(os.path.abspath(__file__))

os.makedirs(f"{path}/models", exist_ok=True)  # Ensure models directory exists
os.makedirs(f"{path}/data", exist_ok=True)  # Ensure data directory exists

import mesa_gym.gyms.grid.staghunt.env as w

VALUE_DIMENSIONS = ["social"]

for value_dim in VALUE_DIMENSIONS:

    env = w.MesaGoalEnv(render_mode=None, map=None, value_dim=value_dim)

    agents = []
    type_agent = {}
    for mesa_agent in env._get_agents():
        agents.append(mesa_agent.unique_id)
        type_agent[mesa_agent.unique_id] = type(mesa_agent).__name__


    from tqdm import tqdm 

    data = {}
    data["fields"] = []

    ######################################
    # Deep q-learning
    ######################################


    def dqn_learning(): 
        from mesa_gym.trainers.DQN import DQNTrainer, device
        import torch
        from itertools import count

        n_episodes = 1000

        replay_batch_size = 32 # 128
        learning_rate = 0.001  # 1e-4
        discount_factor = 0.95 # 0.99
        start_epsilon = 1.0    # 0.9
        epsilon_decay = start_epsilon / (n_episodes / 2)  
        final_epsilon = 0.1    # 0.05
        update_rate = 0.005

        experiment_name = f"DQNlearning_{value_dim}_{n_episodes}_{replay_batch_size}_{update_rate}_{learning_rate}_{discount_factor}_{start_epsilon}_{epsilon_decay}_{final_epsilon}"

        trainers = {}


        for agent in agents:
            trainers[agent] = DQNTrainer(agent=agent,
                                        observation_space=env.observation_space,
                                        action_space=env.action_space[agent],
                                        replay_batch_size=replay_batch_size,
                                        discount_factor=discount_factor,
                                        initial_epsilon=start_epsilon,
                                        final_epsilon=final_epsilon,
                                        epsilon_decay=epsilon_decay,
                                        update_rate=update_rate,
                                        learning_rate=learning_rate
                                        )

        for _ in tqdm(range(n_episodes)):
            observation, info = env.reset()
            states = {agent: torch.tensor(observation[agent], dtype=torch.float32, device=device).unsqueeze(0) for agent in agents}

            for _ in range(500):
                actions = {}
                reward_tensors = {}
                next_states = {}
                
                living_agents = env._get_agents()

                for agent in living_agents:
                    agent_id = agent.unique_id
                    actions[agent_id] = trainers[agent_id].select_action(states[agent_id])

                observation, rewards, terminated, truncated, _ = env.step(actions)
                #print(rewards)

                for agent in living_agents:
                    agent_id = agent.unique_id
                    reward = rewards[agent_id] if agent_id in rewards else 0
                    reward_tensors[agent_id] = torch.tensor([reward], device=device)
                
                done = terminated or truncated

                for agent in living_agents:
                    agent_id = agent.unique_id
                    if terminated:
                        next_states[agent_id] = None
                    if agent_id in observation:
                        next_states[agent_id] = torch.tensor(observation[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                    else:
                        # The agent might have starved or otherwise not present in obs
                        next_states[agent_id] = None
                
                for agent in living_agents:
                    agent_id = agent.unique_id
                    trainers[agent_id].memory.push(states[agent_id], actions[agent_id], next_states[agent_id], reward_tensors[agent_id])
                    states[agent_id] = next_states[agent_id]
                    trainers[agent_id].optimize_model()

                    trainers[agent_id].target_net_state_dict = trainers[agent_id].target_net.state_dict()
                    policy_net_state_dict = trainers[agent_id].policy_net.state_dict()
                    for key in policy_net_state_dict:
                        trainers[agent_id].target_net_state_dict[key] = policy_net_state_dict[key] * update_rate + \
                                                        trainers[agent_id].target_net_state_dict[key] * (1 - update_rate)
                    trainers[agent_id].target_net.load_state_dict(trainers[agent_id].target_net_state_dict)

                if done:
                    break


        # save models

        for agent in agents:
            trainer = trainers[agent]
            filename = os.path.join(path, "models", f"{type_agent[agent]}_{agent}_{experiment_name}.pt")
            torch.save(trainer.policy_net.state_dict(), filename)
            print(f"trained model saved in {filename}")

        return experiment_name, trainers




    experiment_name, trainers = dqn_learning()

    # save data
    filename = f"data/{experiment_name}.pickle"
    with open(f"{path}/{filename}", "wb") as f:
        pickle.dump(data, f)
        print(f"training data saved in {filename}")






######################################
    # q-learning
######################################
MAX_STEPS = 500

def q_learning():
    from mesa_gym.trainers.qlearning import QLearningTrainer

    n_episodes = 10

    learning_rate = 0.01
    discount_factor = 0.95
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    experiment_name = f"qlearning_{value_dim}_{n_episodes}_{learning_rate}_{discount_factor}_{start_epsilon}_{epsilon_decay}_{final_epsilon}"

    trainers = {}
    for agent in agents:
        trainers[agent] = QLearningTrainer(agent=agent, action_space=env.action_space[agent],
                                        learning_rate=learning_rate, initial_epsilon=start_epsilon,
                                        discount_factor=discount_factor, epsilon_decay=epsilon_decay,
                                        final_epsilon=final_epsilon)

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        data[episode] = {}

        step = 0
        while not done and step < MAX_STEPS:
            actions = {}
            for agent in agents:
                actions[agent] = trainers[agent].select_action(obs[agent])
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            # collect data
            data[episode][step] = {}
            for agent in agents:
                data[episode][step][agent] = {}
                data[episode][step][agent]["reward"] = rewards[agent] if agent in rewards else 0
                if agent in info:
                    for key in info[agent]:
                        if key not in data["fields"]:
                            data["fields"].append(key)
                        data[episode][step][agent][key] = info[agent][key]

            # update the agent
            for agent in agents:
                reward = rewards[agent] if agent in rewards else 0
                trainers[agent].update(obs[agent], actions[agent], reward, terminated, next_obs[agent])
            obs = next_obs

            # update if the environment is done and the current obs
            done = terminated or truncated
            step += 1

        for agent in agents:
            trainers[agent].decay_epsilon()

    # save the Q-table

    for trainer in trainers:
        filename = f"models/{type_agent[trainer]}_{trainer}_{experiment_name}.pickle"
        with open(f"{path}/{filename}", "wb") as f:
            pickle.dump(trainers[trainer].q_table(), f)
            print(f"trained model saved in {filename}")

    return experiment_name, trainers


                