from env_AGVsimple_gymnasium import PlantSimAGVsimple

import ray
from ray import tune, air
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air import session

def tune_with_callback():
    tuner = tune.Tuner(
        "DQN",
        #train_function,
        run_config=air.RunConfig(
            local_dir="./trained_models",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_order="max",
                checkpoint_score_attribute="episode_reward_mean",
                num_to_keep=5),
            stop={"episode_reward_mean": 50, "training_iteration": 1000000},
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


if __name__ == '__main__':
    
    # Init.
    def env_creator(env_config):
        return PlantSimAGVsimple()  # return an env instance
    register_env("PlantSimAGVsimple", env_creator)
    ray.init()

    # Configure.
    from ray.rllib.algorithms.dqn.dqn import DQNConfig
    config = DQNConfig().environment(
        env="PlantSimAGVsimple").framework("torch").training(
        replay_buffer_config={"type": "ReplayBuffer", 
                                "capacity": tune.grid_search([50000, 100000, 1000000])}, 
        lr=tune.grid_search([0.0001, 0.0005, 0.001]))


    # Build.
    #algo = config.build()
    tune_with_callback()

    # Test.
    #test_trained_models_DQN()
