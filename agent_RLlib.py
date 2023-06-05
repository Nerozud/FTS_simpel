from env_AGVsimple_multiagent import PlantSimAGVMA
#from env_AGVsimple_gymnasium import PlantSimAGVsimple
import numpy as np
import ray
from ray import tune, air
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import (CombinedStopper, MaximumIterationStopper, TrialPlateauStopper)


stopper = CombinedStopper(
    MaximumIterationStopper(max_iter=1000),
    #TrialPlateauStopper(metric="episode_reward_mean", std=0.2, num_results=100),
)

pbt_ppo = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=50,
    hyperparam_mutations={
        "sgd_minibatch_size": tune.randint(4, 4000),
        "num_sgd_iter": tune.randint(3, 30),
        "clip_param": tune.uniform(0.1, 0.3),
        "lr": tune.uniform(0.000005, 0.003),
        "kl_coeff": tune.uniform(0.3, 1), 
        "kl_target": tune.uniform(0.003, 0.03),
        "gamma": tune.uniform(0.8, 0.9997),
        "lambda_": tune.uniform(0.9, 1),
        "vf_loss_coeff": tune.uniform(0.5, 1),
        "entropy_coeff": tune.uniform(0, 0.01)    
    })

def tune_with_callback():    
    tuner = tune.Tuner(
        "QMIX", # "DQN", "PPO", "QMIX"
        param_space=config,
        tune_config=tune.TuneConfig(
            max_concurrent_trials = 6,
            num_samples = 30,            
            #time_budget_s=3600*24*1, # 1 day
            #scheduler=pbt_ppo,
            # search_alg= BayesOptSearch(metric="episode_reward_mean", 
            #                            mode="max", 
            #                            random_search_steps=0, 
            #                            utility_kwargs={"kind": "ucb", "kappa": 0.5, "xi": 0.0},
            #                            points_to_evaluate=[{"clip_param": 0.1749080237694725, 
            #                                                 "lr": 0.0001789604184437574, 
            #                                                 "kl_coeff": 0.7190609389379257, 
            #                                                 "kl_target": 0.007212503291945786, 
            #                                                 "gamma": 0.9461791901797376, 
            #                                                 "lambda_": 0.9155994520336204, 
            #                                                 "vf_loss_coeff": 0.9330880728874676, 
            #                                                 "entropy_coeff": 0.009507143064099164}],
            #                             verbose=2
            #                             ),
            trial_name_creator=trial_str_creator,
            trial_dirname_creator=trial_str_creator,
            
        ),
        run_config=air.RunConfig(
            local_dir="./trained_models",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=50,
                checkpoint_at_end=True,
                # checkpoint_score_order="max",
                # checkpoint_score_attribute="episode_reward_mean",
                # num_to_keep=5
                ),
            #stop={"episode_reward_mean": 30, "timesteps_total": 1000000},
            stop=stopper,
            callbacks=[WandbLoggerCallback(project="agvs-simple-qmix")]
        )        
    )
    tuner.fit()

def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)

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
                                "capacity": tune.choice([100000, 1000000])}, 
        lr=tune.uniform(1e-4,1e-2),
        gamma=tune.uniform(0.9, 0.999),

        train_batch_size=tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        target_network_update_freq=tune.choice([100, 500, 1000, 2000, 5000, 10000])
        ).exploration(exploration_config={"type": "EpsilonGreedy", 
                            "warmup_timesteps": tune.randint(0, 100000),
                            "epsilon_timesteps": tune.randint(50000, 500000),
                            "final_epsilon": tune.uniform(0.001, 0.01)}
        ).multi_agent(policies={"agv_policy": (None, None, None, {})} ,
                                                           policy_mapping_fn=policy_mapping_fn)
    return config

def get_ppo_multiagent_config():
    from ray.rllib.algorithms.ppo import PPOConfig # .resources(num_gpus=1) müsste GPU aktivieren, funktioniert aber noch nicht
    config = PPOConfig().environment(
        env="PlantSimAGVMA", env_config={"num_agents": 2}
        ).resources(
        #num_gpus=1
        ).framework("torch").training(         
        #horizon=tune.randint(32, 5001), # funktioniert nicht, da horizon nicht in der config ist
        # sgd_minibatch_size=tune.randint(4, 4000),
        sgd_minibatch_size=512,
        # num_sgd_iter=tune.randint(3, 30),
        num_sgd_iter=20,        
        # clip_param=tune.uniform(0.1, 0.3),
        # lr=tune.uniform(0.000005, 0.001),
        # kl_coeff=tune.uniform(0.3, 1), 
        # kl_target=tune.uniform(0.003, 0.03),
        # gamma=tune.uniform(0.8, 0.9997),
        # lambda_=tune.uniform(0.9, 1),
        # vf_loss_coeff=tune.uniform(0.5, 1),
        # entropy_coeff=tune.uniform(0.001, 0.01)
        clip_param=0.1657,
        lr=0.00009462,
        kl_coeff=0.7294, 
        kl_target=0.02122,
        gamma=0.9528,
        lambda_=0.9078,
        vf_loss_coeff=0.9343,
        entropy_coeff=0.009875
        ).multi_agent(  policies={"agv_policy": (None, None, None, {})} ,
                        policy_mapping_fn= policy_mapping_fn)
    return config

def get_qmix_config():
    from ray.rllib.algorithms.qmix import QMixConfig
    config = QMixConfig().environment(
        env="PlantSimAGVMA", env_config={"num_agents": 2, "enable_grouping": True}
    ).framework("torch").training(
        lr=tune.uniform(1e-4,1e-2),
        gamma=tune.uniform(0.9, 0.999),
        train_batch_size=tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        mixer = tune.choice(["qmix", "vdn"]),
        target_network_update_freq=tune.choice([100, 500, 1000, 2000, 5000, 10000]),
        #optim_alpha=tune.uniform(0.1, 0.99)
    ).multi_agent(policies={"agv_policy": (None, None, None, {})} ,
                                                           policy_mapping_fn=policy_mapping_fn)
    return config

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "agv_policy"

if __name__ == '__main__':
    
    # Init.
    # def env_creator(env_config):
    #     return PlantSimAGVMA(env_config)

    def env_creator(env_config):
        env = PlantSimAGVMA(env_config)

        # If agent grouping is enabled in env_config...
        if env_config.get("enable_grouping", False):
            from gymnasium import spaces
            # Get the number of agents from env_config
            num_agents = env_config.get("num_agents", 2)

            # Define agent groups, observation spaces, and action spaces
            groups = {"group_1": [f"agent_{i}" for i in range(num_agents)]}
            group_obs_space = spaces.Tuple(env.observation_space for _ in range(len(groups)))
            group_action_space = spaces.Tuple(env.action_space for _ in range(len(groups)))

            # Group agents
            env = env.with_agent_groups(groups, obs_space=group_obs_space, act_space=group_action_space)

        return env

    register_env("PlantSimAGVMA", env_creator)
    ray.init(object_store_memory=800000000)


    # Configure.
    config = get_qmix_config()

    # Tune. Für Hyperparametersuche mit tune
    tune_with_callback()

    # Resume.
    #tune.run(resume=True, run_or_experiment="DQN")

    # Build & Train. Einfach einen Algorithmus erstellen und trainieren
    # algo = config.build()
    # while True:
    #     print(algo.train())

    # Test.
    # checkpoint_path = "./trained_models/PPO/PPO_1a5e7_00000/checkpoint_001900"
    # test_trained_model(checkpoint_path)
 