import numpy as np
import random


class MultiAgentFilterFactoryEnv:
    """
    Lightweight cooperative multi-agent environment.

    Agents:
        top_agent, bottom_agent, plastic_agent, press_agent, seal_agent

    Each agent chooses:
        0 = idle
        1 = work

    All agents receive the same global reward.
    """

    def __init__(
        self,
        max_steps=100,
        max_inventory=30,
        initial_demand=25,
        breakdown_prob=0.02,
        repair_prob=0.20,
        demand_change_prob=0.05,
    ):
        self.agents = [
            "top_agent",
            "bottom_agent",
            "plastic_agent",
            "press_agent",
            "seal_agent",
        ]

        self.max_steps = max_steps
        self.max_inventory = max_inventory
        self.initial_demand = initial_demand
        self.breakdown_prob = breakdown_prob
        self.repair_prob = repair_prob
        self.demand_change_prob = demand_change_prob

        self.reset()

    def reset(self):
        self.current_step = 0
        self.top = 0
        self.bottom = 0
        self.plastic = 0
        self.semi = 0
        self.finished = 0
        self.demand = self.initial_demand

        self.machine_status = {agent: 1 for agent in self.agents}

        self.invalid_actions = 0
        self.breakdowns = 0
        self.total_wip = 0.0

        return self._get_obs()

    def _get_global_state(self):
        return np.array(
            [
                self.top,
                self.bottom,
                self.plastic,
                self.semi,
                self.finished,
                self.demand,
                self.max_steps - self.current_step,
                self.machine_status["top_agent"],
                self.machine_status["bottom_agent"],
                self.machine_status["plastic_agent"],
                self.machine_status["press_agent"],
                self.machine_status["seal_agent"],
            ],
            dtype=np.float32,
        )

    def _get_obs(self):
        state = self._get_global_state()
        return {agent: state.copy() for agent in self.agents}

    def _apply_dynamic_events(self):
        for agent in self.agents:
            if self.machine_status[agent] == 1:
                if random.random() < self.breakdown_prob:
                    self.machine_status[agent] = 0
                    self.breakdowns += 1
            else:
                if random.random() < self.repair_prob:
                    self.machine_status[agent] = 1

        if random.random() < self.demand_change_prob:
            self.demand = max(0, self.demand + random.choice([-2, -1, 1, 2, 3]))

    def step(self, actions):
        self.current_step += 1
        self._apply_dynamic_events()

        invalid = 0
        finished_now = 0

        if actions["top_agent"] == 1:
            if self.machine_status["top_agent"] == 1 and self.top < self.max_inventory:
                self.top += 1
            else:
                invalid += 1

        if actions["bottom_agent"] == 1:
            if self.machine_status["bottom_agent"] == 1 and self.bottom < self.max_inventory:
                self.bottom += 1
            else:
                invalid += 1

        if actions["plastic_agent"] == 1:
            if self.machine_status["plastic_agent"] == 1 and self.plastic < self.max_inventory:
                self.plastic += 1
            else:
                invalid += 1

        if actions["press_agent"] == 1:
            if (
                self.machine_status["press_agent"] == 1
                and self.top >= 1
                and self.plastic >= 1
                and self.semi < self.max_inventory
            ):
                self.top -= 1
                self.plastic -= 1
                self.semi += 1
            else:
                invalid += 1

        if actions["seal_agent"] == 1:
            if self.machine_status["seal_agent"] == 1 and self.semi >= 1 and self.bottom >= 1:
                self.semi -= 1
                self.bottom -= 1
                self.finished += 1
                finished_now = 1
            else:
                invalid += 1

        self.invalid_actions += invalid

        wip = self.top + self.bottom + self.plastic + self.semi
        self.total_wip += wip

        demand_remaining = max(0, self.demand - self.finished)
        imbalance = abs(self.top - self.plastic) + abs(self.bottom - self.semi)

        reward = 0.0
        reward += 25.0 * finished_now
        reward -= 0.4 * wip
        reward -= 0.1 * demand_remaining
        reward -= 0.1 * imbalance
        reward -= 5.0 * invalid

        done = self.current_step >= self.max_steps

        observations = self._get_obs()
        rewards = {agent: float(reward) for agent in self.agents}
        dones = {agent: done for agent in self.agents}

        info = {
            "finished_filters": int(self.finished),
            "demand": int(self.demand),
            "demand_satisfaction": float(min(1.0, self.finished / max(1, self.demand))),
            "invalid_actions": int(self.invalid_actions),
            "breakdowns": int(self.breakdowns),
            "average_wip": float(self.total_wip / self.current_step),
        }

        return observations, rewards, dones, info
