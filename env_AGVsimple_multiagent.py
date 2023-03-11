from pathlib import Path

import win32com.client as win32
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time

from ray.rllib.env.multi_agent_env import MultiAgentEnv

mod_path = Path(__file__).parent


class PlantSimAGVMA(MultiAgentEnv):
    """ PlantSim AGV Multiagent Environment for OpenAI gymnasium.  """

    metadata = {"render.modes": ["human"]}    
    
    def __init__(self, env_config):
        """Initialisierung der Simulation"""
        super().__init__()
        self.num_agents = env_config["num_agents"]
        #Plant Simulation Initialisierung
        self.PlantSim = win32.Dispatch("Tecnomatix.PlantSimulation.RemoteControl.22.1")
        self.PlantSim.SetLicenseType("Research")
        self.PlantSim.loadmodel ("{}\\simulation_models\\AGVS_simpel_2203.spp".format(mod_path))     

        #self.PlantSim.StartSimulation(".Modelle.Modell.Ereignisverwalter")
        #self.PlantSim.StopSimulation

        #Gesamtzahl der Schritte pro Episode
        self.SchritteProEpisode = 1010
        self.PlantSim.SetValue(".Modelle.Modell.SchritteproEpisode", self.SchritteProEpisode)
        self.PlantSim.ExecuteSimTalk(".Modelle.Modell.Pythonanbindung:=true")
        self.PlantSim.ExecuteSimTalk(f".Modelle.Modell.anzahl_agenten := {self.num_agents}")
        #Zähler für Schritte pro Episode
        self.Schrittanzahl = 0
        
        #Abwarten für vollständingen Ladevorgang
        time.sleep(5)

        # PlantSim-Input definieren: 
            # 1 ID des betrachteten Agenten, beginnend bei 0
            # 2 Station A Belegt, 
            # 3 Station B Belegt, 
            # 4 Puffer1 AnzahlBEs
            # 5 Puffer2 AnzahlBEs
            # 6 AGV 1 XPos,
            # 7 AGV 1 YPos,
            # 8 AGV 1 Geschwindigkeit, 
            # 9 AGV 1 Zielort: 0 = void, 1 = Station A, 2 = Station B, 3 = Senke
            # 10 AGV 1 Inhalt: 0 = beladen, 1 = leer

        # Define the observation shape for each agent
        agent_obs_shape = (5 + 5 * self.num_agents,)

        # Define the observation space as a dictionary mapping agent names to their observations
        # The observations are defined as Box spaces with low and high values for each input
        # The low and high values for the AGV specific inputs are tiled according to the number of agents
        # The shape of each observation is set to the agent_obs_shape defined earlier
        # The dtype is set to np.float32

        self.observation_space = spaces.Box(
                low=np.concatenate([[0, 0, 0, 0, 0], np.tile([100, 120, -1, 0, 0], self.num_agents)]),
                high=np.concatenate([[self.num_agents - 1, 7, 7, 8, 8], np.tile([798, 500, 1, 5, 1], self.num_agents)]),
                shape=agent_obs_shape)
        
        # self.observation_space = gym.spaces.Dict({
        #     f"agent_{i}": spaces.Box(
        #         low=np.concatenate([[0, 0, 0, 0], np.tile([100, 120, -1, 0, 0], self.num_agents)]),
        #         high=np.concatenate([[7, 7, 8, 8], np.tile([798, 500, 1, 3, 1], self.num_agents)]),
        #         shape=agent_obs_shape,
        #         dtype=np.float32
        #     ) for i in range(self.num_agents)
        # })

        # Define action space for each agent
        # 0 Stop, 1 Vorwärts, 2 Rückwärts                
        # self.action_space = gym.spaces.Dict({
        #     f"agent_{i}": spaces.Discrete(3) for i in range(self.num_agents)
        # })
        self.action_space = gym.spaces.Discrete(3)

    def seed (self, seed=None):        
        """Setzen der Zufallszahlengenerator-Seed"""
        self.seed = self.PlantSim.GetValue(".Modelle.Modell.Ereignisverwalter.ZufallszahlenVariante")
        return [seed]

    def step(self, actions):
        """Ausführen einer Aktion und Berechnung des Rewards"""
        self.Schrittanzahl = self.Schrittanzahl + 1
        reward = 0
        done = False
        
        truncated = False

        # durchsatz_senke_alt = self.PlantSim.GetValue(".Modelle.Modell.Senke.StatAnzahlEin")
        # durchsatz_station_a_alt = self.PlantSim.GetValue(".Modelle.Modell.StationA.StatAnzahlEin")
        # durchsatz_station_b_alt = self.PlantSim.GetValue(".Modelle.Modell.StationB.StatAnzahlEin")
        # anzahl_kollisionen_alt = self.PlantSim.GetValue(".Modelle.Modell.anzahl_kollisionen")
    
        # Check that actions are provided for all agents
        assert set(actions.keys()) == set([f"agent_{i}" for i in range(self.num_agents)])

        # Execute actions for each agent
        # Actions: 0 - Angehalten, 1 - Vorwärts, 2 - Rückwärts
        for i, action in enumerate(actions.values()):
            if action == 0:
                self.PlantSim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Angehalten:=true")
            elif action == 1:
                self.PlantSim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Rückwärts:=false")
                self.PlantSim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Angehalten:=false")
            elif action == 2:
                self.PlantSim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Rückwärts:=true")
                self.PlantSim.ExecuteSimTalk(f".BEs.Fahrzeug:{i+1}.Angehalten:=false")


        #Simulationsschritt starten und nach Schrittzeit 1 s wieder stoppen
        self.PlantSim.ExecuteSimTalk(".Modelle.Modell.StepPython")
        
        #neuen State abwarten
        while True:
            if self.PlantSim.GetValue(".Modelle.Modell.Ereignisverwalter.SimulationGestartet") == False:
                break
        
        #neuen State abfragen
        obs = self.get_observation()

        #neuen State auswerten; wenn Durchsatz gestiegen ist, dann richtiges Fahrzeug zuweisen; Kollisionen bestraft derzeitig (noch) alle
        # durchsatz_senke_neu = self.PlantSim.GetValue(".Modelle.Modell.Senke.StatAnzahlEin")
        
        # durchsatz_senke_änderung_agent_id = -1
        # durchsatz_station_a_änderung_agent_id = -1
        # durchsatz_station_b_änderung_agent_id = -1

        # if durchsatz_senke_neu > durchsatz_senke_alt:
        #     durchsatz_senke_änderung_agent_id = self.PlantSim.GetValue(".Modelle.Modell.senke_letztes_fahrzeugID") - 1
        # durchsatz_station_a_neu = self.PlantSim.GetValue(".Modelle.Modell.StationA.StatAnzahlEin")
        # if durchsatz_station_a_neu > durchsatz_station_a_alt: 
        #     durchsatz_station_a_änderung_agent_id = self.PlantSim.GetValue(".Modelle.Modell.stationA_letztes_fahrzeugID") - 1
        # durchsatz_station_b_neu = self.PlantSim.GetValue(".Modelle.Modell.StationB.StatAnzahlEin")
        # if durchsatz_station_b_neu > durchsatz_station_b_alt: 
        #     durchsatz_station_b_änderung_agent_id = self.PlantSim.GetValue(".Modelle.Modell.stationB_letztes_fahrzeugID") - 1
        # anzahl_kollisionen_neu = self.PlantSim.GetValue(".Modelle.Modell.anzahl_kollisionen")

        #Reward berechnen
        rewards = {}
        for i in range(self.num_agents):
            # reward = 0
            # if i == durchsatz_senke_änderung_agent_id:
            #     reward += durchsatz_senke_neu - durchsatz_senke_alt            
            # if i == durchsatz_station_a_änderung_agent_id:
            #     reward += (durchsatz_station_a_neu - durchsatz_station_a_alt) * 0.1
            # if i == durchsatz_station_b_änderung_agent_id:
            #     reward += (durchsatz_station_b_neu - durchsatz_station_b_alt) * 0.1
            # reward -= (anzahl_kollisionen_neu - anzahl_kollisionen_alt)

            reward = self.PlantSim.GetValue(f".Modelle.Modell.agent{i+1}_reward")
        
            rewards[f"agent_{i}"] = reward

        #Simulation beendet?
        self.simulation_over = self.PlantSim.GetValue(".Modelle.Modell.Episode_beendet")
        if self.simulation_over:
            done = {"__all__": True}
            truncated = {"__all__": self.Schrittanzahl < self.SchritteProEpisode}
        else:
            done = {"__all__": False}
            truncated = {"__all__": False}


        # set the info dictionary
        info = {}
        # info = {"durchsatz_senke": durchsatz_senke_neu,
        #         "durchsatz_station_a": durchsatz_station_a_neu,
        #         "durchsatz_station_b": durchsatz_station_b_neu,
        #         "anzahl_kollisionen": anzahl_kollisionen_neu}

        return obs, rewards, done, truncated, info

    def get_observation(self):
        obs = {}       
        
        state_mapping = {"Wartend": 0, "Arbeitend": 1, "Blockiert": 2, "Rüstend": 3,
                        "Gestört": 4, "Angehalten": 5, "Pausiert": 6, "Ungeplant": 7}
        station_a_val = state_mapping.get(self.PlantSim.GetValue(".Modelle.Modell.StationA.ResMomentanZustand"), -1)
        station_b_val = state_mapping.get(self.PlantSim.GetValue(".Modelle.Modell.StationB.ResMomentanZustand"), -1)
        puffer1_val = self.PlantSim.GetValue(".Modelle.Modell.Puffer1.AnzahlBEs")
        puffer2_val = self.PlantSim.GetValue(".Modelle.Modell.Puffer2.AnzahlBEs")
        
        zielort_mapping = {"": 0, "*.Modelle.Modell.StationA": 1, "*.Modelle.Modell.StationB": 2, "*.Modelle.Modell.Senke": 3, "*.Modelle.Modell.Puffer1": 4, "*.Modelle.Modell.Puffer2": 5} # mit * ist korrekt

        for i in range(self.num_agents):
            # Create a list of observations for all agents
            obs_list = [i,
                        station_a_val,
                        station_b_val,
                        puffer1_val,
                        puffer2_val]
            # Append the observations for each agent to the list
            for j in range(self.num_agents): 
                obs_list.append(self.PlantSim.GetValue(f".BEs.Fahrzeug:{j+1}.XPos"))
                obs_list.append(self.PlantSim.GetValue(f".BEs.Fahrzeug:{j+1}.YPos"))
                obs_list.append(self.PlantSim.GetValue(f".BEs.Fahrzeug:{j+1}.Momentangeschw"))
                zielort_value = self.PlantSim.GetValue(f".BEs.Fahrzeug:{j+1}.Zielort")
                if zielort_value is None: #Wenn Zielort kein String, leeren String draus machen
                    zielort_value = ""        
                zielort = zielort_mapping.get(zielort_value if len(zielort_value) > 0 else "", -1) #Wenn Zielort leer ist, dann "" als Key verwenden
                obs_list.append(zielort)
                obs_list.append(self.PlantSim.GetValue(f".BEs.Fahrzeug:{j+1}.leer"))


            # Convert the list to a numpy array and assign it to the agent's key
            obs[f"agent_{i}"] = np.array(obs_list)
        return obs
   

    def reset(self, *, seed=None, options=None):
        """Reset the state of the environment and returns an initial observation."""
        # Call the parent class's reset() method to initialize the simulation
        super().reset(seed=seed)
        
        # Reset the step counter
        self.Schrittanzahl = 0

        # Reset the simulation
        self.PlantSim.ResetSimulation(".Modelle.Modell.Ereignisverwalter")
        
        # Set a new random seed for the simulation
        self.PlantSim.ExecuteSimTalk(f".Modelle.Modell.Ereignisverwalter.ZufallszahlenVariante := {np.random.randint(0, 1000000)}")

        # Wait for all vehicles to be removed from the simulation
        while True:
            if self.PlantSim.GetValue(".BEs.Fahrzeug.AnzahlKinder") == 0:
                break
        
        # Start the simulation to spawn vehicles
        self.PlantSim.StartSimulation(".Modelle.Modell.Ereignisverwalter")
        self.PlantSim.StopSimulation

        # Wait for all vehicles to be spawned in the simulation
        while True:
            if self.PlantSim.GetValue(".BEs.Fahrzeug.AnzahlKinder") == self.num_agents:
                break
        
        obs = self.get_observation()

        info = {}
        #print ("Reset-State: ", self.state)

        # Return the initial observations and info for all agents
        return obs, info


    def render(self):
        """Renders the environment."""
        self.PlantSim.SetVisible(True)


    def close(self):
        """Closes the environment."""
        #self.PlantSim.CloseModel()
        self.PlantSim.Quit()
