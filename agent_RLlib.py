from env_AGVsimple_multiagent import PlantSimAGVMA
#from env_AGVsimple_gymnasium import PlantSimAGVsimple
import numpy as np
import ray
from ray import tune, air
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.bayesopt import BayesOptSearch


def tune_with_callback():    
    tuner = tune.Tuner(
        "DQN",
        tune_config=tune.TuneConfig(
            #max_concurrent_trials = 3,
            num_samples = 1,
            #search_alg= BayesOptSearch(metric="episode_reward_mean", mode="max")
        ),
        run_config=air.RunConfig(
            local_dir="./trained_models",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_order="max",
                checkpoint_score_attribute="episode_reward_mean",
                num_to_keep=5),
            stop={"episode_reward_mean": 30, "timesteps_total": 3000000},
            callbacks=[WandbLoggerCallback(project="agvs-simple-dqn-hyperopt")]
        ),
        param_space=config
    )
    tuner.fit()

def test_trained_model(checkpoint_path, num_episodes=10):
    from ray.rllib.algorithms.algorithm import Algorithm

    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Get the policy using the policy_id
    policy_id = "agv_policy"
    env = env_creator({"num_agents": 2})
    env.render()
    for i in range(num_episodes):        
        obs = env.reset()[0]
        print(f"Initial observation: {obs}")
        done = {'__all__': False}
        episode_reward = 0        
        while not done['__all__']:
            """ Compute actions for each agent """
            actions = dict()
            #print("Type of actions:", type(actions))
            for j in range(2):
                actions[f"agent_{j}"] = algo.compute_single_action(obs[f"agent_{j}"], policy_id=policy_id, explore=False)
            #print(f"Actions: {actions}")
                
            obs, rewards, done, truncated, info = env.step(actions)
            #print(f"done: {done}, rewards: {rewards}")
            episode_reward += sum(rewards.values())
        
        print(f"Episode {i + 1} reward: {episode_reward}")



def get_dqn_multiagent_config():
    from ray.rllib.algorithms.dqn.dqn import DQNConfig
    config = DQNConfig().environment(
        env="PlantSimAGVMA", env_config={"num_agents": 2}).framework("torch").training(
        replay_buffer_config={"type": "ReplayBuffer", 
                                "capacity": tune.grid_search([100000, 1000000])}, 
        lr=tune.grid_search([0.00005, 0.0001, 0.0005])).multi_agent(policies={"agv_policy": (None, None, None, {})} ,
                                                           policy_mapping_fn=policy_mapping_fn)
    return config

def get_ppo_multiagent_config():
    from ray.rllib.algorithms.ppo import PPOConfig # .resources(num_gpus=1) müsste GPU aktivieren, funktioniert aber noch nicht
    config = PPOConfig().environment(
        env="PlantSimAGVMA", env_config={"num_agents": 2}).framework("torch").training(         
        #horizon=tune.randint(32, 5001), # funktioniert nicht, da horizon nicht in der config ist
        sgd_minibatch_size=tune.randint(4, 4000),
        num_sgd_iter=tune.randint(3, 30),
        clip_param=tune.uniform(0.1, 0.3),
        lr=tune.uniform(0.000005, 0.003),
        kl_coeff=tune.uniform(0.3, 1), 
        kl_target=tune.uniform(0.003, 0.03),
        gamma=tune.uniform(0.8, 0.9997),
        lambda_=tune.uniform(0.9, 1),
        vf_loss_coeff=tune.uniform(0.5, 1),
        entropy_coeff=tune.uniform(0, 0.01)
        ).multi_agent(  policies={"agv_policy": (None, None, None, {})} ,
                        policy_mapping_fn= policy_mapping_fn)
    return config

def get_rainbow_config():
    from ray.rllib.algorithms.dqn.dqn import DQNConfig
    config = DQNConfig().environment(
        env="PlantSimAGVsimple").framework("torch").training(
        n_step=tune.grid_search([1, 3, 5]),
        noisy=True,
        num_atoms=tune.grid_search([1, 4, 8]),
        v_min=tune.grid_search([-100.0, -10.0, -1.0]),
        v_max=tune.grid_search([100.0, 10.0, 1.0]),
        lr=0.0001)
    return config

def get_rainbow_one_run_config():
    from ray.rllib.algorithms.dqn.dqn import DQNConfig
    config = DQNConfig().environment(
        env="PlantSimAGVsimple").framework("torch").training(
        n_step=3,
        noisy=True,
        num_atoms= 4,
        v_min= -10.0,
        v_max= 10.0,
        lr=0.0001)
    return config

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "agv_policy"

if __name__ == '__main__':
    
    # Init.
    def env_creator(env_config):
        return PlantSimAGVMA(env_config)

    register_env("PlantSimAGVMA", env_creator)
    ray.init(object_store_memory=800000000)


    # Configure.
    config = get_dqn_multiagent_config()

    # Tune. Für Hyperparametersuche mit tune
    tune_with_callback()

    # Resume.
    #tune.run(resume=True, run_or_experiment="DQN")

    # Build & Train. Einfach einen Algorithmus erstellen und trainieren
    # algo = config.build()
    # while True:
    #     print(algo.train())

    # Test.
    # checkpoint_path = "./trained_models/DQN/DQN_PlantSimAGVMA_ac230_00004_4_lr=0.0001,capacity=1000000_2023-03-20_11-13-03/checkpoint_003000"
    # test_trained_model(checkpoint_path)
 