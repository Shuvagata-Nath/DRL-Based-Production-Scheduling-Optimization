import random
import pickle
from collections import defaultdict
from multi_agent_factory_env import MultiAgentFilterFactoryEnv


def discretize_obs(obs):
    top, bottom, plastic, semi, finished, demand, remaining = obs[:7]

    def bucket(x):
        if x <= 0:
            return 0
        if x <= 2:
            return 1
        if x <= 5:
            return 2
        return 3

    return (
        bucket(top),
        bucket(bottom),
        bucket(plastic),
        bucket(semi),
        bucket(demand - finished),
        bucket(remaining),
    )


class IndependentQLearningAgents:
    def __init__(self, agents, alpha=0.1, gamma=0.95, epsilon=1.0):
        self.agents = agents
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_tables = {
            agent: defaultdict(lambda: [0.0, 0.0])
            for agent in agents
        }

    def choose_actions(self, observations):
        actions = {}

        for agent in self.agents:
            state = discretize_obs(observations[agent])

            if random.random() < self.epsilon:
                action = random.choice([0, 1])
            else:
                q_values = self.q_tables[agent][state]
                action = int(q_values[1] > q_values[0])

            actions[agent] = action

        return actions

    def update(self, observations, actions, rewards, next_observations):
        for agent in self.agents:
            state = discretize_obs(observations[agent])
            next_state = discretize_obs(next_observations[agent])
            action = actions[agent]
            reward = rewards[agent]

            old_q = self.q_tables[agent][state][action]
            next_max = max(self.q_tables[agent][next_state])

            new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
            self.q_tables[agent][state][action] = new_q


def main():
    env = MultiAgentFilterFactoryEnv()
    learners = IndependentQLearningAgents(env.agents)

    episodes = 5000

    for episode in range(episodes):
        observations = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            actions = learners.choose_actions(observations)
            next_observations, rewards, dones, info = env.step(actions)

            learners.update(observations, actions, rewards, next_observations)

            observations = next_observations
            done = all(dones.values())
            total_reward += list(rewards.values())[0]

        learners.epsilon = max(0.05, learners.epsilon * 0.995)

        if (episode + 1) % 500 == 0:
            print(
                f"Episode {episode + 1}, "
                f"Reward={total_reward:.2f}, "
                f"Finished={info['finished_filters']}, "
                f"DemandSat={info['demand_satisfaction']:.2f}, "
                f"Epsilon={learners.epsilon:.3f}"
            )

    serializable_q_tables = {
        agent: dict(table)
        for agent, table in learners.q_tables.items()
    }

    with open("multi_agent_iql.pkl", "wb") as f:
        pickle.dump(serializable_q_tables, f)

    print("Multi-agent IQL model saved as multi_agent_iql.pkl")


if __name__ == "__main__":
    main()
