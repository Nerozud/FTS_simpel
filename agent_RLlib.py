""" Train and test a multi-agent reinforcement learning model using Ray RLlib. """

import os

import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.stopper import (
    CombinedStopper,
    MaximumIterationStopper,
    TrialPlateauStopper,
)
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.qmix import QMixConfig


from ray.rllib.algorithms.algorithm import Algorithm
import torch


from env_agv_simple_pomdp import PlantSimAGVMA

# from env_AGVsimple_multiagent import PlantSimAGVMA
# from env_AGVsimple_gymnasium import PlantSimAGVsimple


stopper = CombinedStopper(
    MaximumIterationStopper(max_iter=300),
    # TrialPlateauStopper(metric="episode_reward_mean", std=0.2, num_results=100),
)

pbt_ppo = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=50,
    hyperparam_mutations={
        # "sgd_minibatch_size": tune.randint(4, 4000),
        # "num_sgd_iter": tune.randint(3, 30),
        "clip_param": tune.uniform(0.1, 0.3),
        "lr": tune.uniform(0.000005, 0.003),
        "kl_coeff": tune.uniform(0.3, 1),
        "kl_target": tune.uniform(0.003, 0.03),
        "gamma": tune.uniform(0.8, 0.9997),
        "lambda_": tune.uniform(0.9, 1),
        "vf_loss_coeff": tune.uniform(0.5, 1),
        "entropy_coeff": tune.uniform(0, 0.01),
    },
)


def tune_with_callback():
    """
    Tune with callback for logging to wandb
    """
    tuner = tune.Tuner(
        "PPO",  # "DQN", "PPO", "QMIX", "SAC"
        param_space=config,
        tune_config=tune.TuneConfig(
            max_concurrent_trials=1,
            num_samples=10,
            # time_budget_s=3600*24*1, # 1 day
            # scheduler=pbt_ppo,
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
            local_dir=os.path.abspath("./trained_models"),
            storage_path=os.path.abspath("./trained_models"),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=50,
                checkpoint_at_end=False,
                checkpoint_score_order="max",
                checkpoint_score_attribute="episode_reward_mean",
                num_to_keep=5,
            ),
            # stop={"episode_reward_mean": 30, "timesteps_total": 1000000},
            stop=stopper,
            callbacks=[WandbLoggerCallback(project="agvs-simple-ppo-test")],
        ),
    )
    tuner.fit()


def trial_str_creator(trial):
    """
    Create a string for the trial name and directory name
    """
    return f"{trial.trainable_name}_{trial.trial_id}"


def test_trained_model(checkpoint_path, num_episodes=10):
    """
    Test a trained model with a given checkpoint path
    """

    # Initialize the RLlib Algorithm from a checkpoint.
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # Create your custom environment.
    env = env_creator({"num_agents": 2})
    env.render()

    for episode in range(num_episodes):
        obs = env.reset()[0]
        done = {"__all__": False}
        episode_reward = 0

        state_list = {agent_id: [torch.zeros(64), torch.zeros(64)] for agent_id in obs}
        initial_state_list = state_list

        actions = {agent_id: 0 for agent_id in obs}
        rewards = {agent_id: 0.0 for agent_id in obs}

        next_state_list = {}

        while not done["__all__"]:
            for j in range(2):
                actions[f"agent_{j}"], next_state_list[f"agent_{j}"], _ = (
                    algo.compute_single_action(
                        obs[f"agent_{j}"],
                        state=state_list[f"agent_{j}"],
                        policy_id="agv_policy",
                        prev_action=actions[f"agent_{j}"],
                        prev_reward=rewards[f"agent_{j}"],
                        explore=True,
                    )
                )  # for PG algorithms true
            # Step the environment.
            obs, rewards, done, truncated, info = env.step(actions)
            episode_reward += sum(rewards.values())

            # Update states for all agents.
            state_list = next_state_list

        state_list = initial_state_list
        print(f"Episode {episode + 1} reward: {episode_reward}")


def get_dqn_multiagent_config():
    """Get the DQN multiagent config"""
    config = (
        DQNConfig()
        .environment(env="PlantSimAGVMA", env_config={"num_agents": 2})
        .framework("torch")
        .training(
            replay_buffer_config={
                "type": "ReplayBuffer",
                "capacity": tune.choice([100000, 1000000]),
            },
            lr=tune.uniform(1e-4, 1e-2),
            gamma=tune.uniform(0.9, 0.999),
            train_batch_size=tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
            target_network_update_freq=tune.choice([100, 500, 1000, 2000, 5000, 10000]),
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "warmup_timesteps": tune.randint(0, 100000),
                "epsilon_timesteps": tune.randint(50000, 500000),
                "final_epsilon": tune.uniform(0.001, 0.01),
            }
        )
        .multi_agent(
            policies={"agv_policy": (None, None, None, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
    )
    return config


def get_ppo_multiagent_config():
    """Get the PPO multiagent config"""

    config = (
        PPOConfig()
        .environment(
            env="PlantSimAGVMA", env_config={"num_agents": 2, "enable_grouping": False}
        )
        .resources(
            num_gpus=1,
            # num_gpus_per_learner_worker=0.2,
        )
        .framework(
            "torch"
            # "tf2"
        )
        .training(
            # sgd_minibatch_size=tune.grid_search([1024, 2048, 4000]),
            train_batch_size=25000,
            sgd_minibatch_size=25000,
            # num_sgd_iter=tune.randint(3, 30),
            num_sgd_iter=10,
            model={
                "fcnet_hiddens": [64, 64],
                "use_lstm": True,
                "lstm_cell_size": 64,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True,
            },
            #         model={"fcnet_hiddens": tune.grid_search([[64, 64], [128, 128]]),
            #             #"_disable_preprocessor_api": True,
            #             "use_lstm": tune.grid_search([True, False]),
            #             "lstm_cell_size": 64,
            #             "lstm_use_prev_action": True,
            #             "lstm_use_prev_reward": True,
            #             "vf_share_layers": tune.grid_search([True, False]),
            # },
            # clip_param=tune.grid_search([0.05, 0.2]),
            # lr=tune.grid_search([0.0001, 0.0003, 0.0005]),
            # kl_coeff=tune.uniform(0.3, 1),
            # kl_target=tune.uniform(0.003, 0.03),
            # gamma=tune.uniform(0.8, 0.9997),
            # lambda_=tune.uniform(0.9, 1),
            # vf_loss_coeff=tune.uniform(0.5, 1),
            # entropy_coeff=tune.grid_search([0.001, 0.003, 0.01]),
            clip_param=0.2,
            lr=0.0005,
            # kl_coeff=0.5,
            # kl_target=0.01,
            gamma=0.99,
            lambda_=0.95,
            # vf_loss_coeff=0.9,
            entropy_coeff=0.01,
            optimizer={
                "adam_epsilon": 1e-5,
                # "beta1": 0.99,
                # "beta2": 0.99,
            },
        )
        .multi_agent(
            policies={"agv_policy": (None, None, None, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    return config


def get_sac_multiagent_config():
    config = (
        SACConfig()
        .environment(
            env="PlantSimAGVMA", env_config={"num_agents": 2, "enable_grouping": False}
        )
        .training(
            train_batch_size=1024,
            # train_batch_size=tune.grid_search([256, 4000]),
        )
        .multi_agent(
            policies={"agv_policy": (None, None, None, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
    )
    return config


def get_qmix_config():
    config = (
        QMixConfig()
        .environment(
            env="PlantSimAGVMA", env_config={"num_agents": 2, "enable_grouping": True}
        )
        .framework("torch")
        .training(
            lr=tune.uniform(1e-4, 1e-2),
            gamma=tune.uniform(0.9, 0.999),
            train_batch_size=tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
            mixer=tune.choice(["qmix", "vdn"]),
            target_network_update_freq=tune.choice([100, 500, 1000, 2000, 5000, 10000]),
            # optim_alpha=tune.uniform(0.1, 0.99),
        )
        .multi_agent(
            policies={"agv_policy": (None, None, None, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
    )
    return config


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map an agent to a policy."""
    return "agv_policy"


if __name__ == "__main__":

    # Init.
    def env_creator(env_config):
        """Create an environment instance."""
        return PlantSimAGVMA(env_config)

    # def env_creator(env_config):
    #     """Create an environment instance."""
    #     env = PlantSimAGVMA(env_config)

    #     # If agent grouping is enabled in env_config...
    #     if env_config.get("enable_grouping", False):

    #         # Get the number of agents from env_config
    #         num_agents = env_config.get("num_agents")

    #         # Define agent groups, observation spaces, and action spaces
    #         groups = {"group_1": [f"agent_{i}" for i in range(num_agents)]}
    #         group_obs_space = spaces.Tuple(
    #             env.observation_space for _ in range(num_agents)
    #         )
    #         group_action_space = spaces.Tuple(
    #             env.action_space for _ in range(num_agents)
    #         )

    #         # Group agents
    #         env = env.with_agent_groups(
    #             groups, obs_space=group_obs_space, act_space=group_action_space
    #         )
    #     return env

    register_env("PlantSimAGVMA", env_creator)
    ray.init()

    # Configure.
    config = get_ppo_multiagent_config()

    # config ["batch_mode"] = "complete_episodes"

    config["num_envs_per_worker"] = 4
    config["num_env_runners"] = 2
    config["num_rollout_workers"] = 8
    config["restart_failed_sub_environments"] = True

    # config["monitor"] = True

    # Tune. Für Hyperparametersuche mit tune
    tune_with_callback()

    # Resume.
    # tune.run(restore="trained_models\PPO_2024-04-16_14-24-57\PPO_56862_00002\checkpoint_000019",
    #          storage_path=os.path.abspath("./trained_models"),
    #          run_or_experiment="PPO",
    #          config=config,
    #          stop={"training_iteration": 2000},
    #          num_samples=3,
    #          callbacks=[WandbLoggerCallback(project="agvs-simple-ppo-test")],
    #          checkpoint_config=air.CheckpointConfig(
    #             checkpoint_frequency=50,
    #             checkpoint_at_end=False,
    #             checkpoint_score_order="max",
    #             checkpoint_score_attribute="episode_reward_mean",
    #             num_to_keep=5
    #             ),
    #          )

    # Build & Train. Einfach einen Algorithmus erstellen und trainieren
    # algo = config.build()
    # while True:
    #     print(algo.train())

    # Test.
    # checkpoint_path = "trained_models\PPO_2024-04-22_15-44-30\PPO_PlantSimAGVMA_719c2_00001_1_2024-04-22_15-44-30\checkpoint_000017"

    # checkpoint_path = "trained_models\PPO_2024-05-21_18-15-13\PPO_4dee8_00000\checkpoint_000005" # good, 2 agents
    # checkpoint_path = "trained_models\PPO_2024-05-13_11-27-15\PPO_fc969_00001\checkpoint_000005" # local optimum, 3 agents
    # test_trained_model(checkpoint_path)
