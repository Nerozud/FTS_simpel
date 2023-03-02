from env_AGVsimple_multiagent import PlantSimAGVMA

import ray
from ray import tune, air
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback

def tune_with_callback():
    tuner = tune.Tuner(
        "DQN",
        tune_config=tune.TuneConfig(
            #max_concurrent_trials = 6,
            num_samples = 1,
        ),
        run_config=air.RunConfig(
            local_dir="./trained_models",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_order="max",
                checkpoint_score_attribute="episode_reward_mean",
                num_to_keep=5),
            stop={"episode_reward_mean": 30, "timesteps_total": 1500000},
            callbacks=[WandbLoggerCallback(project="agvs-simple")]
        ),
        param_space=config
    )
    tuner.fit()

def test_trained_models_DQN():
    """ Test the trained model."""
    # Load the trained model.
    #experiment_path = "./trained_models/DQN/DQN_PlantSimAGVsimple_042ec_00000_0_lr=0.0001_2023-02-17_14-57-29"
    #print(f"Loading results from {experiment_path}...")
    #restored_tuner = tune.Tuner.restore(experiment_path) 
    
    # Get the best result based on a particular metric.
    #best_result = restored_tuner.get_best_result(metric="episode_reward_mean", mode="max")

    # Get the best checkpoint corresponding to the best result.
    #best_checkpoint = best_result.checkpoint
    best_checkpoint = "./trained_models/DQN/DQN_PlantSimAGVsimple_042ec_00000_0_lr=0.0001_2023-02-17_14-57-29/checkpoint_000124"
    print(f"Best checkpoint: {best_checkpoint}")
    
    # Load the algorithm from the best checkpoint.
    from ray.rllib.algorithms.algorithm import Algorithm
    algo = Algorithm.from_checkpoint(best_checkpoint)
    
    # Test the trained model.
    env = env_creator({})
    episode_reward = 0
    done = False
    obs = env.reset()[0]
    env.render()
    while not done:
        action = algo.compute_single_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
    print(f"Episode reward: {episode_reward}")


def get_dqn_config():
    from ray.rllib.algorithms.dqn.dqn import DQNConfig
    config = DQNConfig().environment(
        env="PlantSimAGVsimple").framework("torch").training(
        # replay_buffer_config={"type": "ReplayBuffer", 
        #                         "capacity": tune.grid_search([50000, 100000, 1000000])}, 
        lr=tune.grid_search([0.00001, 0.00005, 0.0001, 0.0005]))        
    return config

def get_dqn_multiagent_config():
    from ray.rllib.agents.dqn.dqn import DQNConfig
    config = DQNConfig().environment(
        env="PlantSimAGVMA",
        env_config={"num_agents": 2}).framework("torch").training(
        # replay_buffer_config={"type": "ReplayBuffer", 
        #                         "capacity": tune.grid_search([50000, 100000, 1000000])}, 
        lr=tune.grid_search([0.0001, 0.0005])).multiagent(policies={
            "policy_0": (None, env_creator({}).observation_space, env_creator({}).action_space, {}),
            "policy_1": (None, env_creator({}).observation_space, env_creator({}).action_space, {}),
        }, policy_mapping_fn=lambda agent_id: agent_id)
                
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

if __name__ == '__main__':
    
    # Init.
    def env_creator(env_config):
        return PlantSimAGVMA()  # return an env instance
    register_env("PlantSimAGVMA", env_creator)
    ray.init()

    # Configure.
    config = get_dqn_multiagent_config()

    # Tune. Für Hyperparametersuche mit tune
    tune_with_callback()

    # Build & Train. Einfach einen Algorithmus erstellen und trainieren
    # algo = config.build()
    # while True:
    #     print(algo.train())

    # Test.
    #test_trained_models_DQN()
