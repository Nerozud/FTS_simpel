"""plant_sim AGV Multiagent Environment for OpenAI gymnasium."""

from pathlib import Path

import gymnasium as gym

import numpy as np
import win32com.client as win32

from ray.rllib.env.multi_agent_env import MultiAgentEnv

mod_path = Path(__file__).parent


class PlantSimAGVMA(MultiAgentEnv):
    """PlantSim AGV Multiagent Environment for OpenAI gymnasium."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config):
        """Initialisierung der Simulation"""
        super().__init__()
        self.env_config = env_config
        self.num_agents = env_config["num_agents"]
        self._agent_ids = {f"agent_{i}" for i in range(self.num_agents)}
        # Plant Simulation Initialisierung
        self.plant_sim = win32.Dispatch("Tecnomatix.PlantSimulation.RemoteControl.22.1")
        self.plant_sim.SetLicenseType("Research")
        self.plant_sim.loadmodel(f"{mod_path}\\simulation_models\\AGVS_simpel_2311.spp")

        # self.plant_sim.StartSimulation(".Modelle.Modell.Ereignisverwalter")
        # self.plant_sim.StopSimulation

        self.step_count = 0
        self.steps_per_episode = 2000
        self.plant_sim.SetValue(
            ".Modelle.Modell.SchritteproEpisode", self.steps_per_episode
        )
        self.plant_sim.ExecuteSimTalk(".Modelle.Modell.Pythonanbindung:=true")
        self.plant_sim.ExecuteSimTalk(
            f".Modelle.Modell.anzahl_agenten := {self.num_agents}"
        )

        # plant_sim-Input definieren:
        # 1 ID of considered AGV, starting with 0
        # 2 AGV XPos,  --> Object ID of location
        # 3 AGV YPos, --> relative position on path
        # 4 AGV Geschwindigkeit,
        # 5 AGV Zielort: 0 = void, 1 = Station A, 2 = Station B, 3 = Senke, 4 = Puffer1, 5 = Puffer2
        # 6 AGV Inhalt: 0 = beladen, 1 = leer
        # 7 Distance to target
        # 8 AGV X Distance to next AGV
        # 9 AGV Y Distance to next AGV

        # self.observation_space = spaces.MultiDiscrete([self.num_agents, 799, 501, 3, 6, 2, 100, 700, 400])
        # self.observation_space = gym.spaces.MultiDiscrete([40, 25, 3, 6, 2, 60, 80, 40])
        # self.observation_space = gym.spaces.MultiDiscrete([40, 25, 3, 6, 2, 60, 40, 25])

        self.observation_space = gym.spaces.Dict(
            {
                "position": gym.spaces.MultiDiscrete([800, 500]),
                "speed": gym.spaces.Discrete(3),
                "target": gym.spaces.Discrete(6),
                "load_status": gym.spaces.Discrete(2),
                "route_length": gym.spaces.Discrete(60),
                "nearest_distances": gym.spaces.MultiDiscrete([800, 400]),
            }
        )

        # Define action space
        # 0 stop, 1 forwards, 2 backwards
        self.action_space = gym.spaces.Discrete(3)

    def seed(self, seed=None):
        """Get the seed for the environment."""
        seed = self.plant_sim.GetValue(
            ".Modelle.Modell.Ereignisverwalter.ZufallszahlenVariante"
        )
        return seed

    def step(self, action_dict):
        """Execute one step in the environment."""
        self.step_count = self.step_count + 1
        reward = 0
        done = False
        truncated = False

        # Execute actions for each agent
        # Actions: 0 - Angehalten, 1 - Vorwärts, 2 - Rückwärts
        for i, action in enumerate(action_dict.values()):
            if action == 0:
                self.plant_sim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Angehalten:=true")
            elif action == 1:
                self.plant_sim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Rückwärts:=false")
                self.plant_sim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Angehalten:=false")
            elif action == 2:
                self.plant_sim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Rückwärts:=true")
                self.plant_sim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Angehalten:=false")

        # Simulationsschritt starten und nach Schrittzeit 1 s wieder stoppen
        self.plant_sim.ExecuteSimTalk(".Modelle.Modell.StepPython")

        # neuen State abwarten
        while True:
            if (
                self.plant_sim.GetValue(
                    ".Modelle.Modell.Ereignisverwalter.SimulationGestartet"
                )
                == False
            ):
                break

        obs = self.get_observation()

        # calculate rewards
        rewards = {}
        for i in range(self.num_agents):
            reward = self.plant_sim.GetValue(f".Modelle.Modell.agent{i+1}_reward")

            rewards[f"agent_{i}"] = reward

        # simulation finished?
        simulation_over = self.plant_sim.GetValue(".Modelle.Modell.Episode_beendet")
        if simulation_over:
            done = {"__all__": True}
            truncated = {"__all__": self.step_count < self.steps_per_episode}
        else:
            done = {"__all__": False}
            truncated = {"__all__": False}

        info = {}

        return obs, rewards, done, truncated, info

    def get_observation(self):
        """Get the observation for all agents."""

        obs = {}
        positions = {}
        zielort_mapping = {
            "": 0,
            "*.Modelle.Modell.StationA": 1,
            "*.Modelle.Modell.StationB": 2,
            "*.Modelle.Modell.Senke": 3,
            "*.Modelle.Modell.Puffer1": 4,
            "*.Modelle.Modell.Puffer2": 5,
        }  # mit * ist korrekt

        # Collect positions of all agents
        for i in range(self.num_agents):
            x_pos = self.plant_sim.GetValue(f".BEs.Fahrzeug:{i+1}.XPos")
            y_pos = self.plant_sim.GetValue(f".BEs.Fahrzeug:{i+1}.YPos")
            positions[i] = (x_pos, y_pos)

        for i in range(self.num_agents):
            x_pos, y_pos = positions[i]

            geschwindigkeit = self.plant_sim.GetValue(
                f".BEs.Fahrzeug:{i+1}.Momentangeschw"
            )
            if geschwindigkeit == 0:
                geschwindigkeits_kategorie = 0
            elif geschwindigkeit > 0:
                geschwindigkeits_kategorie = 1
            else:  # Geschwindigkeit < 0
                geschwindigkeits_kategorie = 2

            zielort_value = self.plant_sim.GetValue(f".BEs.Fahrzeug:{i+1}.Zielort")
            if (
                zielort_value is None
            ):  # Wenn Zielort kein String, leeren String draus machen
                zielort_value = ""
            zielort = zielort_mapping.get(
                zielort_value if len(zielort_value) > 0 else "", -1
            )

            min_dist = float("inf")
            nearest_x = 0
            nearest_y = 0
            for j in positions:
                if i != j:
                    other_x, other_y = positions[j]
                    distance = np.sqrt((x_pos - other_x) ** 2 + (y_pos - other_y) ** 2)
                    if distance < min_dist:
                        min_dist = distance
                        nearest_x = abs(other_x - x_pos)
                        nearest_y = abs(other_y - y_pos)

            obs[f"agent_{i}"] = {
                "position": np.array([x_pos, y_pos], dtype=np.uint16),
                "speed": np.array(geschwindigkeits_kategorie, dtype=np.uint8),
                "target": np.array(zielort, dtype=np.uint8),
                "load_status": np.array(
                    self.plant_sim.GetValue(f".BEs.Fahrzeug:{i+1}.leer"), dtype=np.uint8
                ),
                "route_length": np.array(
                    round(
                        self.plant_sim.GetValue(
                            f".Modelle.Modell.agent{i+1}_route_length"
                        )
                        + 1
                    ),
                    dtype=np.uint8,
                ),
                "nearest_distances": np.array([nearest_x, nearest_y], dtype=np.uint16),
            }

        return obs

    def reset(self, *, seed=None, options=None):
        """Reset the state of the environment and returns an initial observation."""
        # Call the parent class's reset() method to initialize the simulation
        super().reset(seed=seed)

        # Reset the step counter
        self.step_count = 0

        # Reset the simulation
        self.plant_sim.ResetSimulation(".Modelle.Modell.Ereignisverwalter")

        # Set a new random seed for the simulation
        self.plant_sim.ExecuteSimTalk(
            f".Modelle.Modell.Ereignisverwalter.ZufallszahlenVariante := {np.random.randint(0, 1000000)}"
        )

        # Wait for all vehicles to be removed from the simulation
        while True:
            if self.plant_sim.GetValue(".BEs.Fahrzeug.AnzahlKinder") == 0:
                break

        # Start the simulation to spawn vehicles
        self.plant_sim.StartSimulation(".Modelle.Modell.Ereignisverwalter")
        self.plant_sim.StopSimulation()

        # Wait for all vehicles to be spawned in the simulation
        while True:
            if self.plant_sim.GetValue(".BEs.Fahrzeug.AnzahlKinder") == self.num_agents:
                break

        obs = self.get_observation()

        # print("reset observation types:")
        # for agent_id, ob in obs.items():
        #     print(f"Agent {agent_id} observation type: {type(ob)}, dtype: {ob.dtype}")

        info = {}
        # print ("Reset-State: ", self.state)

        # Return the initial observations and info for all agents
        # return {agent_id: obs[agent_id] for agent_id in self._agent_ids}, info
        return obs, info

    def render(self):
        """Renders the environment."""
        self.plant_sim.SetVisible(True)

    def close(self):
        """Closes the environment."""
        self.plant_sim.CloseModel()
        self.plant_sim.Quit()

    def observation_space_sample(self):
        """Returns a sample from the observation space."""
        return {
            agent_id: self.observation_space.sample() for agent_id in self._agent_ids
        }

    def observation_space_contains(self, obs):
        """Check if `obs` is in the observation space."""
        return all(self.observation_space.contains(o) for o in obs.values())

    def action_space_sample(self, unbekannteszweitesargument):
        # Generiert eine zufällige Aktion für den spezifischen Agenten basierend auf seiner Agenten-ID
        return {agent_id: self.action_space.sample() for agent_id in self._agent_ids}

    def action_space_contains(self, action):
        """Check if `action` is in the action space."""
        return all(self.action_space.contains(a) for a in action.values())
