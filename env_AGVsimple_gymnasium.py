from pathlib import Path

import win32com.client as win32
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time


mod_path = Path(__file__).parent
#print(mod_path)
#print(("{}\\simulation_models\\AGV_simpel_15_1.spp".format(mod_path)))

class PlantSimAGVsimple(gym.Env):
    """Einfaches AGV-Beispiel mit PlantSimulation und OpenAI Gym-Standard"""

    metadata = {"render.modes": ["human"]}    
    
    def __init__(self):
        """Initialisierung der Simulation"""
        super().__init__()
        
        #Plant Simulation Initialisierung
        self.PlantSim = win32.Dispatch("Tecnomatix.PlantSimulation.RemoteControl.22.1")
        self.PlantSim.SetLicenseType("Research")
        self.PlantSim.loadmodel ("{}\\simulation_models\\AGV_simpel_2201.spp".format(mod_path))     

        #self.PlantSim.StartSimulation(".Modelle.Modell.Ereignisverwalter")
        #self.PlantSim.StopSimulation

        #Gesamtzahl der Schritte pro Episode
        self.SchritteProEpisode = 1010
        self.PlantSim.SetValue(".Modelle.Modell.SchritteproEpisode", self.SchritteProEpisode)
        #Zähler für Schritte pro Episode
        self.Schrittanzahl = 0
        
        #Abwarten für vollständingen Ladevorgang
        time.sleep(5)

        # PlantSim-Input definieren: Maschine 1 Belegt, Maschine 2 Belegt, AGV Buchungs-Position, AGV Geschwindigkeit, Anzahl der BEs auf AGV
        #self.observation_space = np.array([0, 0, 0, 0, 0])
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 14, 1, 6], dtype=np.float32), 
            shape=(5, ), 
            dtype=np.float32
        )
        # 0 Stop, 1 Vorwärts, 2 Rückwärts
        self.action_space = spaces.Discrete(3)

    def seed (self, seed=None):        
        """Setzen der Zufallszahlengenerator-Seed"""
        self.seed = self.PlantSim.GetValue(".Modelle.Modell.Ereignisverwalter.ZufallszahlenVariante")
        return [seed]

    def step(self, action):
        """Ausführen einer Aktion und Berechnung des Rewards"""
        self.Schrittanzahl = self.Schrittanzahl + 1
        reward = 0
        done = False
        truncated = False

        alterWeg = self.PlantSim.GetValue(".BEs.Fahrzeug:1.StatWegstrecke")
        alterDurchsatz = self.PlantSim.GetValue(".Modelle.Modell.Einzelstation2.StatAnzahlEin")

        # Actions: 0 - Angehalten, 1 - Vorwärts, 2 - Rückwärts
        if action == 0:
            self.PlantSim.ExecuteSimTalk(".BEs.Fahrzeug:1.Angehalten:=true")
        if action == 1:
            self.PlantSim.ExecuteSimTalk(".BEs.Fahrzeug:1.Rückwärts:=false")
            self.PlantSim.ExecuteSimTalk(".BEs.Fahrzeug:1.Angehalten:=false")
        if action == 2:
            self.PlantSim.ExecuteSimTalk(".BEs.Fahrzeug:1.Rückwärts:=true")
            self.PlantSim.ExecuteSimTalk(".BEs.Fahrzeug:1.Angehalten:=false")

        #Simulationsschritt starten und nach Schrittzeit 1 s wieder stoppen
        self.PlantSim.ExecuteSimTalk(".Modelle.Modell.StepPython")
        
        #neuen State abwarten
        while True:
            if self.PlantSim.GetValue(".Modelle.Modell.Ereignisverwalter.SimulationGestartet") == False:
                break
        
        neuerWeg = self.PlantSim.GetValue(".BEs.Fahrzeug:1.StatWegstrecke")
        neuerDurchsatz = self.PlantSim.GetValue(".Modelle.Modell.Einzelstation2.StatAnzahlEin")
        Wegänderung = neuerWeg - alterWeg
        Durchsatzänderung = neuerDurchsatz - alterDurchsatz

        reward = Durchsatzänderung - Wegänderung * 0.01
        #reward = Durchsatzänderung

        self.simulation_over = self.PlantSim.GetValue(".Modelle.Modell.Episode_beendet")
        if self.simulation_over:
            done = True
            truncated = (self.Schrittanzahl < self.SchritteProEpisode)

        self.state = [self.PlantSim.GetValue(".Modelle.Modell.Einzelstation1.Belegt"), 
                      self.PlantSim.GetValue(".Modelle.Modell.Einzelstation2.Belegt"), 
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.BuchPos"), 
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.Momentangeschw"),
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.AnzahlBEs")]
        
        return np.array(self.state).astype(np.float32), reward, done, truncated, {}
        

    def reset(self, *, seed=None, options=None):
        """Reset the state of the environment and returns an initial observation."""
        super().reset(seed=seed)
        self.Schrittanzahl = 0
        
        self.PlantSim.ResetSimulation(".Modelle.Modell.Ereignisverwalter")
        
        # Zufallszahlen ggf. noch abhängig von Episodenanzahl machen??
        self.PlantSim.ExecuteSimTalk(f".Modelle.Modell.Ereignisverwalter.ZufallszahlenVariante := {np.random.randint(0, 1000000)}")
        #self.PlantSim.SetValue(".Modelle.Modell.Ereignisverwalter", np.random.randint(0, 1000000))

        #Abwarten bis Fahrzeug nicht mehr existiert (nach einer Episode) - relevant bei Render = True
        while True:            
            if self.PlantSim.GetValue(".BEs.Fahrzeug.AnzahlKinder") == 0:
               break
        #Simulation starten, um Fahrzeuge zu erzeugen
        self.PlantSim.StartSimulation(".Modelle.Modell.Ereignisverwalter")
        self.PlantSim.StopSimulation
        #Abwarten bis Fahrzeug existiert
        while True:            
            if self.PlantSim.GetValue(".BEs.Fahrzeug.AnzahlKinder") == 1:
               break
        self.state = [self.PlantSim.GetValue(".Modelle.Modell.Einzelstation1.Belegt"), 
                      self.PlantSim.GetValue(".Modelle.Modell.Einzelstation2.Belegt"), 
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.BuchPos"), 
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.Momentangeschw"),
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.AnzahlBEs",)]

        info = {}
        #print ("Reset-State: ", self.state)
        return np.array(self.state).astype(np.float32), info


    def render(self):
        """Renders the environment."""
        self.PlantSim.SetVisible(True)


    def close(self):
        """Closes the environment."""
        #self.PlantSim.CloseModel()
        self.PlantSim.Quit()
