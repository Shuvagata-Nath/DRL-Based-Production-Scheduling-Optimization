from dynamic_factory_env import DynamicFilterFactoryEnv
from multi_agent_factory_env import MultiAgentFilterFactoryEnv


def test_centralized():
    env = DynamicFilterFactoryEnv()
    obs, _ = env.reset()

    print("Centralized dynamic environment test")
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Action:", action, "Reward:", reward, "Info:", info)


def test_multi_agent():
    env = MultiAgentFilterFactoryEnv()
    obs = env.reset()

    print("\nMulti-agent environment test")
    for _ in range(5):
        actions = {agent: 1 for agent in env.agents}
        obs, rewards, dones, info = env.step(actions)
        print("Reward:", list(rewards.values())[0], "Info:", info)


if __name__ == "__main__":
    test_centralized()
    test_multi_agent()
