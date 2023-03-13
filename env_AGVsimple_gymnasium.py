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
        self.PlantSim.loadmodel ("{}\\simulation_models\\AGVS_simpel_2201.spp".format(mod_path))     

        #self.PlantSim.StartSimulation(".Modelle.Modell.Ereignisverwalter")
        #self.PlantSim.StopSimulation

        #Gesamtzahl der Schritte pro Episode
        self.SchritteProEpisode = 1810
        self.PlantSim.SetValue(".Modelle.Modell.SchritteproEpisode", self.SchritteProEpisode)
        #self.PlantSim.SetValue(".Modelle.Modell.PythonAnbindung", 1)
        self.PlantSim.ExecuteSimTalk(".Modelle.Modell.Pythonanbindung:=true")
        #Zähler für Schritte pro Episode
        self.Schrittanzahl = 0
        
        #Abwarten für vollständingen Ladevorgang
        time.sleep(5)

        # PlantSim-Input definieren: 
            # 1 Station A Belegt, 
            # 2 Station B Belegt, 
            # 3 Puffer1 AnzahlBEs
            # 4 Puffer2 AnzahlBEs
            # 5 AGV 1 XPos,
            # 6 AGV 1 YPos,
            # 7 AGV 1 Geschwindigkeit, 
            # 8 AGV 1 Zielort: 0 = void, 1 = Station A, 2 = Station B, 3 = Senke
            # 9 AGV 1 Inhalt: 0 = beladen, 1 = leer

        #self.observation_space = np.array([0, 0, 0, 0, 0])
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 100, 120, -1, 0, 0], dtype=np.float32),
            high=np.array([7, 7, 8, 8, 798, 500, 1, 3, 1], dtype=np.float32), 
            shape=(9, ), 
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

        durchsatz_senke_alt = self.PlantSim.GetValue(".Modelle.Modell.Senke.StatAnzahlEin")
        durchsatz_station_a_alt = self.PlantSim.GetValue(".Modelle.Modell.StationA.StatAnzahlEin")
        durchsatz_station_b_alt = self.PlantSim.GetValue(".Modelle.Modell.StationB.StatAnzahlEin")

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
        
        #neuen State abfragen

        state_mapping = {"Wartend": 0, "Arbeitend": 1, "Blockiert": 2, "Rüstend": 3,
                        "Gestört": 4, "Angehalten": 5, "Pausiert": 6, "Ungeplant": 7}
        station_a_val = state_mapping.get(self.PlantSim.GetValue(".Modelle.Modell.StationA.ResMomentanZustand"), -1)
        station_b_val = state_mapping.get(self.PlantSim.GetValue(".Modelle.Modell.StationB.ResMomentanZustand"), -1)

        zielort_mapping = {"": 0, "*.Modelle.Modell.StationA": 1, "*.Modelle.Modell.StationB": 2, "*.Modelle.Modell.Senke": 3}
        zielort_value = self.PlantSim.GetValue(".BEs.Fahrzeug:1.Zielort")
        if zielort_value is None: #Wenn Zielort kein String, leeren String draus machen
            zielort_value = ""        
        zielort = zielort_mapping.get(zielort_value if len(zielort_value) > 0 else "", -1) #Wenn Zielort leer ist, dann "" als Key verwenden
        #print ("Zielort Get Value: ", self.PlantSim.GetValue(".BEs.Fahrzeug:1.Zielort"), "Zielort Mapping: ", zielort_value, zielort)

        self.state = [station_a_val, 
                      station_b_val,
                      self.PlantSim.GetValue(".Modelle.Modell.Puffer1.AnzahlBEs"),
                      self.PlantSim.GetValue(".Modelle.Modell.Puffer2.AnzahlBEs"),
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.XPos"),
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.YPos"), 
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.Momentangeschw"),
                      zielort,
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.leer")]    

        durchsatz_senke_neu = self.PlantSim.GetValue(".Modelle.Modell.Senke.StatAnzahlEin")
        durchsatz_station_a_neu = self.PlantSim.GetValue(".Modelle.Modell.StationA.StatAnzahlEin")
        durchsatz_station_b_neu = self.PlantSim.GetValue(".Modelle.Modell.StationB.StatAnzahlEin")

        #Reward berechnen
        durchsatz_aenderung = durchsatz_senke_neu - durchsatz_senke_alt \
                            + (durchsatz_station_a_neu - durchsatz_station_a_alt) * 0.1 \
                            + (durchsatz_station_b_neu - durchsatz_station_b_alt) * 0.1
        reward = durchsatz_aenderung

        #Simulation beendet?
        self.simulation_over = self.PlantSim.GetValue(".Modelle.Modell.Episode_beendet")
        if self.simulation_over:
            done = True
            truncated = (self.Schrittanzahl < self.SchritteProEpisode)
      
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
        #Simulation starten, um Fahrzeuge zu erzeugen, folgende Befehle können verbessert werden
        self.PlantSim.StartSimulation(".Modelle.Modell.Ereignisverwalter")
        self.PlantSim.StopSimulation
        #Abwarten bis Fahrzeug existiert
        while True:            
            if self.PlantSim.GetValue(".BEs.Fahrzeug.AnzahlKinder") == 1:
               break

        state_mapping = {"Wartend": 0, "Arbeitend": 1, "Blockiert": 2, "Rüstend": 3,
                        "Gestört": 4, "Angehalten": 5, "Pausiert": 6, "Ungeplant": 7}
        station_a_val = state_mapping.get(self.PlantSim.GetValue(".Modelle.Modell.StationA.ResMomentanZustand"), -1)
        station_b_val = state_mapping.get(self.PlantSim.GetValue(".Modelle.Modell.StationB.ResMomentanZustand"), -1)

        zielort_mapping = {"": 0, "*.Modelle.Modell.StationA": 1, "*.Modelle.Modell.StationB": 2, "*.Modelle.Modell.Senke": 3}
        zielort_value = self.PlantSim.GetValue(".BEs.Fahrzeug:1.Zielort")
        if zielort_value is None: #Wenn Zielort kein String, leeren String draus machen
            zielort_value = ""        
        zielort = zielort_mapping.get(zielort_value if len(zielort_value) > 0 else "", -1) #Wenn Zielort leer ist, dann "" als Key verwenden
        #print ("Zielort: ", self.PlantSim.GetValue(".BEs.Fahrzeug:1.Zielort"))
        self.state = [station_a_val, 
                      station_b_val,
                      self.PlantSim.GetValue(".Modelle.Modell.Puffer1.AnzahlBEs"),
                      self.PlantSim.GetValue(".Modelle.Modell.Puffer2.AnzahlBEs"),
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.XPos"),
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.YPos"), 
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.Momentangeschw"),
                      zielort,
                      self.PlantSim.GetValue(".BEs.Fahrzeug:1.leer")]     

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
