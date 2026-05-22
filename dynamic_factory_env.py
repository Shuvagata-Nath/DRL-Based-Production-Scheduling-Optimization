import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class DynamicFilterFactoryEnv(gym.Env):
    """
    Centralized single-agent factory environment with dynamic stochastic events.

    State:
    [
        top_cloth, bottom_cloth, plastic_ring, semi_finished, finished_filters,
        demand, remaining_time,
        top_machine_working, bottom_machine_working, plastic_machine_working,
        press_machine_working, seal_machine_working
    ]

    Actions:
    0 = idle
    1 = cut top cloth
    2 = cut bottom cloth
    3 = mold plastic ring
    4 = press
    5 = seal
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps=100,
        max_inventory=30,
        initial_demand=25,
        breakdown_prob=0.02,
        repair_prob=0.20,
        demand_change_prob=0.05,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.max_inventory = max_inventory
        self.initial_demand = initial_demand
        self.breakdown_prob = breakdown_prob
        self.repair_prob = repair_prob
        self.demand_change_prob = demand_change_prob

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(12,),
            dtype=np.float32,
        )

        self.state = None
        self.current_step = 0
        self.invalid_actions = 0
        self.breakdown_count = 0
        self.total_wip = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.invalid_actions = 0
        self.breakdown_count = 0
        self.total_wip = 0.0

        self.state = np.array(
            [0, 0, 0, 0, 0, self.initial_demand, self.max_steps, 1, 1, 1, 1, 1],
            dtype=np.float32,
        )

        return self.state.copy(), {}

    def _apply_dynamic_events(self):
        # Machine statuses are stored at indices 7 to 11.
        for i in range(7, 12):
            if self.state[i] == 1:
                if random.random() < self.breakdown_prob:
                    self.state[i] = 0
                    self.breakdown_count += 1
            else:
                if random.random() < self.repair_prob:
                    self.state[i] = 1

        if random.random() < self.demand_change_prob:
            self.state[5] = max(0, self.state[5] + random.choice([-2, -1, 1, 2, 3]))

    def step(self, action):
        action = int(action)
        self.current_step += 1
        self._apply_dynamic_events()

        reward = 0.0
        invalid = False
        finished_now = 0

        if action == 0:
            reward -= 1.0

        elif action == 1:
            if self.state[7] == 1 and self.state[0] < self.max_inventory:
                self.state[0] += 1
            else:
                invalid = True

        elif action == 2:
            if self.state[8] == 1 and self.state[1] < self.max_inventory:
                self.state[1] += 1
            else:
                invalid = True

        elif action == 3:
            if self.state[9] == 1 and self.state[2] < self.max_inventory:
                self.state[2] += 1
            else:
                invalid = True

        elif action == 4:
            if (
                self.state[10] == 1
                and self.state[0] >= 1
                and self.state[2] >= 1
                and self.state[3] < self.max_inventory
            ):
                self.state[0] -= 1
                self.state[2] -= 1
                self.state[3] += 1
            else:
                invalid = True

        elif action == 5:
            if self.state[11] == 1 and self.state[3] >= 1 and self.state[1] >= 1:
                self.state[3] -= 1
                self.state[1] -= 1
                self.state[4] += 1
                finished_now = 1
            else:
                invalid = True

        if invalid:
            self.invalid_actions += 1
            reward -= 10.0

        wip = self.state[0] + self.state[1] + self.state[2] + self.state[3]
        self.total_wip += wip

        demand_remaining = max(0, self.state[5] - self.state[4])
        imbalance = abs(self.state[0] - self.state[2]) + abs(self.state[1] - self.state[3])

        reward += 25.0 * finished_now
        reward -= 0.4 * wip
        reward -= 0.1 * demand_remaining
        reward -= 0.1 * imbalance

        self.state[6] = self.max_steps - self.current_step

        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "finished_filters": int(self.state[4]),
            "demand": int(self.state[5]),
            "demand_satisfaction": float(min(1.0, self.state[4] / max(1, self.state[5]))),
            "invalid_actions": int(self.invalid_actions),
            "breakdowns": int(self.breakdown_count),
            "average_wip": float(self.total_wip / self.current_step),
        }

        return self.state.copy(), float(reward), terminated, truncated, info

    def render(self):
        print(
            f"Top={int(self.state[0])}, Bottom={int(self.state[1])}, "
            f"Plastic={int(self.state[2])}, Semi={int(self.state[3])}, "
            f"Finished={int(self.state[4])}, Demand={int(self.state[5])}, "
            f"Machines={self.state[7:12].astype(int).tolist()}"
        )
