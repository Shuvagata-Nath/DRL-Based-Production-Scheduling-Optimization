from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from dynamic_factory_env import DynamicFilterFactoryEnv


def main():
    env = DynamicFilterFactoryEnv()
    check_env(env, warn=True)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0005,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.98,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        verbose=1,
    )

    model.learn(total_timesteps=100000)
    model.save("dynamic_centralized_dqn")
    print("Centralized dynamic DQN model saved as dynamic_centralized_dqn.zip")


if __name__ == "__main__":
    main()
