import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from dynamic_factory_env import DynamicFilterFactoryEnv
from multi_agent_factory_env import MultiAgentFilterFactoryEnv
from train_multi_agent_iql import discretize_obs


def evaluate_centralized_dqn(episodes=100):
    model = DQN.load("dynamic_centralized_dqn")
    records = []

    for ep in range(episodes):
        env = DynamicFilterFactoryEnv()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        final_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward
            final_info = info

        records.append({
            "method": "Centralized DQN",
            "episode": ep,
            "finished_filters": final_info["finished_filters"],
            "total_reward": total_reward,
            "invalid_actions": final_info["invalid_actions"],
            "breakdowns": final_info["breakdowns"],
            "average_wip": final_info["average_wip"],
            "demand_satisfaction": final_info["demand_satisfaction"],
        })

    return records


def evaluate_multi_agent_iql(episodes=100):
    with open("multi_agent_iql.pkl", "rb") as f:
        q_tables = pickle.load(f)

    records = []

    for ep in range(episodes):
        env = MultiAgentFilterFactoryEnv()
        obs = env.reset()
        done = False
        total_reward = 0.0
        final_info = None

        while not done:
            actions = {}

            for agent in env.agents:
                state = discretize_obs(obs[agent])
                q_values = q_tables[agent].get(state, [0.0, 0.0])
                actions[agent] = int(q_values[1] > q_values[0])

            obs, rewards, dones, info = env.step(actions)
            done = all(dones.values())
            total_reward += list(rewards.values())[0]
            final_info = info

        records.append({
            "method": "Multi-Agent IQL",
            "episode": ep,
            "finished_filters": final_info["finished_filters"],
            "total_reward": total_reward,
            "invalid_actions": final_info["invalid_actions"],
            "breakdowns": final_info["breakdowns"],
            "average_wip": final_info["average_wip"],
            "demand_satisfaction": final_info["demand_satisfaction"],
        })

    return records


def greedy_multi_agent_actions(env):
    actions = {
        "top_agent": 0,
        "bottom_agent": 0,
        "plastic_agent": 0,
        "press_agent": 0,
        "seal_agent": 0,
    }

    if env.semi > 0 and env.bottom > 0:
        actions["seal_agent"] = 1

    if env.top > 0 and env.plastic > 0:
        actions["press_agent"] = 1

    if env.top <= env.plastic:
        actions["top_agent"] = 1

    if env.plastic <= env.top:
        actions["plastic_agent"] = 1

    if env.bottom <= env.semi + 1:
        actions["bottom_agent"] = 1

    return actions


def evaluate_greedy_multi_agent(episodes=100):
    records = []

    for ep in range(episodes):
        env = MultiAgentFilterFactoryEnv()
        obs = env.reset()
        done = False
        total_reward = 0.0
        final_info = None

        while not done:
            actions = greedy_multi_agent_actions(env)
            obs, rewards, dones, info = env.step(actions)
            done = all(dones.values())
            total_reward += list(rewards.values())[0]
            final_info = info

        records.append({
            "method": "Greedy Multi-Agent",
            "episode": ep,
            "finished_filters": final_info["finished_filters"],
            "total_reward": total_reward,
            "invalid_actions": final_info["invalid_actions"],
            "breakdowns": final_info["breakdowns"],
            "average_wip": final_info["average_wip"],
            "demand_satisfaction": final_info["demand_satisfaction"],
        })

    return records


def evaluate_random_multi_agent(episodes=100):
    records = []

    for ep in range(episodes):
        env = MultiAgentFilterFactoryEnv()
        obs = env.reset()
        done = False
        total_reward = 0.0
        final_info = None

        while not done:
            actions = {agent: random.choice([0, 1]) for agent in env.agents}
            obs, rewards, dones, info = env.step(actions)
            done = all(dones.values())
            total_reward += list(rewards.values())[0]
            final_info = info

        records.append({
            "method": "Random Multi-Agent",
            "episode": ep,
            "finished_filters": final_info["finished_filters"],
            "total_reward": total_reward,
            "invalid_actions": final_info["invalid_actions"],
            "breakdowns": final_info["breakdowns"],
            "average_wip": final_info["average_wip"],
            "demand_satisfaction": final_info["demand_satisfaction"],
        })

    return records


def main():
    all_records = []
    all_records += evaluate_centralized_dqn()
    all_records += evaluate_multi_agent_iql()
    all_records += evaluate_greedy_multi_agent()
    all_records += evaluate_random_multi_agent()

    df = pd.DataFrame(all_records)
    df.to_csv("advanced_evaluation_results.csv", index=False)

    summary = df.groupby("method").agg({
        "finished_filters": ["mean", "std"],
        "total_reward": ["mean", "std"],
        "invalid_actions": ["mean", "std"],
        "breakdowns": ["mean", "std"],
        "average_wip": ["mean", "std"],
        "demand_satisfaction": ["mean", "std"],
    })

    print(summary)
    summary.to_csv("advanced_summary_results.csv")

    for metric in [
        "finished_filters",
        "total_reward",
        "invalid_actions",
        "average_wip",
        "demand_satisfaction",
    ]:
        plt.figure()
        df.groupby("method")[metric].mean().plot(kind="bar")
        plt.title(f"Average {metric} by method")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f"advanced_{metric}_comparison.png")
        plt.close()


if __name__ == "__main__":
    main()
