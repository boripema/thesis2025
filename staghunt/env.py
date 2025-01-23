import mesa_gym.gyms.grid.staghunt.world as w
import numpy as np
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces


class MesaGoalEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, map=None, value_dim=None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.fps = self.metadata["render_fps"]

        self.value_dim = value_dim

        self.map = map
        self.model = self._get_world()

        self.booting = True

        self.potential_actions = w.AgentBody.get_directions()
        n_actions = len(self.potential_actions)

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()

        self.entities = self._get_entities()
        self.agents = self._get_agents()
        self.events = {agent.unique_id: {} for agent in self.agents}
        self.info = {}

        size = self.model.width * self.model.height

        for agent in self.agents:
            self.action_space[agent.unique_id] = spaces.Discrete(n_actions)

        MIN = 0; MAX = 1
        n_features = 53
        features_high = np.array([MAX] * n_features, dtype=np.float32)
        features_low = np.array([MIN] * n_features, dtype=np.float32)
        self.observation_space = spaces.Box(features_low, features_high)

    
    def _get_world(self):
        # Use the predefined map if provided
        if self.map is not None:
            return w.load_world(self.map)

        # Otherwise, generate a random map
        width = 6
        height = 6
        return w.create_random_world(
            width, height, {
                w.HunterA: 1,
                w.HunterB: 1,
                w.Stag: 6,
            }
        )
        
    def _get_entities(self):
        return self.model.entities
        
    def _get_agents(self):
        return [e for e in self.model.entities if isinstance(e, w.AgentBody) and e.is_alive]

    
    def _get_obs(self):
        observations = {}
        for agent in self._get_agents():
            observations[agent.unique_id] = agent.get_percepts()
        return observations

    
    
    def _get_info(self, events):
        info = self.info

        for agent in self._get_agents():
            if agent.unique_id not in info:
                info[agent.unique_id] = {}

        for agent, event in events:
            if event not in info[agent.unique_id]:
                info[agent.unique_id][event] = 0
            info[agent.unique_id][event] += 1

        return info

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)


        if self.booting:
            self.booting = False
        else:
            self.model = self._get_world()

        if self.render_mode == "human":
            self.view = w.WorldView(self.model, self.fps)
            self.view.init()
            self.view.show()

        observation = self._get_obs()
        self.info = {}

        return observation, self.info
    
    def _get_rewards(self, events):
        
        rewards = {}
        
        self.events = {agent.unique_id: 0 for agent in self.agents}

        for agent, event in events:

            if self.value_dim == "hunger":
                if event == "hunted":
                    rewards[agent.unique_id] = 1

            elif self.value_dim == "sustainability":

                if event == "hunted":
                    if self.model.nstags == 2:
                        rewards[agent.unique_id] = -1
                    elif self.model.nstags == 1:
                        rewards[agent.unique_id] = -5
                    elif self.model.nstags == 0:
                        rewards[agent.unique_id] = -100


            elif self.value_dim == "social":
                living_agents = self._get_agents()

                # If there's only one agent left, or none, skip the social logic
                if len(living_agents) < 2:
                    return rewards
                
                other_agent = living_agents[1 - agent.unique_id]

                if event == "hunted":
                    if agent.satiety == 30 and other_agent.satiety < 15:
                        rewards[agent.unique_id] = -1
                    if agent.satiety > 25 and other_agent.satiety < 10:
                        rewards[agent.unique_id] = -1
                    if agent.satiety > 20 and other_agent.satiety < 5:
                        rewards[agent.unique_id] = 15 - agent.satiety
                    else:
                        rewards[agent.unique_id] = 0.01
                
                if event == "starved":
                    rewards[other_agent.unique_id] = -other_agent.satiety
        
        return rewards

        """
            other_agent = living_agents[1 - agent.unique_id]

            # Simple threshold to decide if the other agent "desperately needs" the stag
            CRITICAL_SATIETY = 5

            if event == "hunted":
                # If the other agent's satiety is very low, punish "selfish" hunting
                if other_agent.satiety < CRITICAL_SATIETY and agent.satiety > CRITICAL_SATIETY:
                    rewards[agent.unique_id] = -5.0
                elif other_agent.satiety < agent.satiety:
                    rewards[agent.unique_id] = -0.5
                else: 
                    rewards[agent.unique_id] = 1

            elif self.value_dim == "social":
                living_agents = self._get_agents()
                if len(living_agents) < 2:
                    return rewards

                nstags = max(self.model.nstags, 1)
                other_agent = living_agents[1 - agent.unique_id]

                if event == "hunted":
                    if agent.satiety > other_agent.satiety:
                        rewards[agent.unique_id] = -1
                    else:
                        rewards[agent.unique_id] = 1

                elif event == "starved":
                    rewards[agent.unique_id] = -5
                    rewards[other_agent.unique_id] = -other_agent.satiety
        """
    
    def step(self, actions):
        
        for agent in self._get_agents():
            agent.next_action = self.potential_actions[actions[agent.unique_id]]

        terminated, events = self.model.step()
        
        if terminated:
            terminated = True # ensure global termination

        if self.render_mode == "human":
            self._render_frame()

        rewards = self._get_rewards(events)
        observations = self._get_obs()
        info = self._get_info(events)
        
        return observations, rewards, terminated, False, info
    
    def render(self):
        self.view.show()

    def _render_frame(self):
        self.view.show()