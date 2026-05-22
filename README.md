#  Microfiber Mesh Factory Environment - Reinforcement Learning Project

## Overview

This project models a multi-stage industrial filter manufacturing line as a reinforcement learning problem.
The system simulates production flow, machine coordination, inventory buildup, stochastic failures, and changing customer demand inside a manufacturing environment.

Two different control approaches are implemented:

* Centralized Deep Q-Network (DQN)
* Decentralized Multi-Agent Independent Q-Learning (IQL)

The project also includes heuristic and random baselines for comparison.

---

## Features

### Manufacturing Simulation

* Multi-stage production pipeline
* Inventory and work-in-progress (WIP) tracking
* Dynamic customer demand
* Machine breakdown and repair events
* Inventory capacity constraints

### Reinforcement Learning

* Centralized single-agent DQN scheduler
* Cooperative multi-agent Independent Q-Learning
* Shared reward coordination
* Stochastic environment interaction

### Evaluation Framework

The project compares:

* Centralized DQN
* Multi-Agent IQL
* Greedy heuristic policy
* Random baseline

Metrics:

* Finished products
* Total reward
* Invalid actions
* Demand satisfaction
* Average WIP
* Machine breakdown statistics

---

## Project Structure

```text
dynamic_factory_env.py
multi_agent_factory_env.py
train_centralized_dqn_dynamic.py
train_multi_agent_iql.py
evaluate_advanced.py
test_envs.py
requirements.txt
README.md
```

---

## Environment Description

### Production Stages

1. Top cloth cutting
2. Bottom cloth cutting
3. Plastic ring molding
4. Pressing
5. Sealing

Finished filters are produced only when all required intermediate components are available.

---

## Dynamic Events

The simulation includes random industrial events such as:

* Machine failures
* Machine repairs
* Demand fluctuations
* Inventory overflow limits
* Work-in-progress penalties

These events make the scheduling problem non-deterministic and closer to realistic manufacturing systems.

---

## Centralized RL Formulation

The centralized environment uses a single agent that controls the entire production line.

### Actions

```text
0 = idle
1 = cut top cloth
2 = cut bottom cloth
3 = mold plastic ring
4 = press
5 = seal
```

The centralized agent is trained using Stable-Baselines3 DQN.

---

## Multi-Agent RL Formulation

The decentralized environment contains five cooperative agents:

* top_agent
* bottom_agent
* plastic_agent
* press_agent
* seal_agent

Each agent chooses:

```text
0 = idle
1 = work
```

All agents receive the same global reward to encourage cooperative production behavior.

---

## Running the Project

Test environments:

```bash
python test_envs.py
```

Train centralized DQN:

```bash
python train_centralized_dqn_dynamic.py
```

Train multi-agent IQL:

```bash
python train_multi_agent_iql.py
```

Run evaluation:

```bash
python evaluate_advanced.py
```

---
## Experimental Findings and Discussion

The evaluation results show that the multi-agent approaches achieved higher production output than the centralized DQN model, but the comparison also reveals an important design difference between the environments. In the centralized DQN setup, the agent selects only one production action per timestep. In the multi-agent setup, five agents can act at the same timestep, allowing multiple production stages to operate in parallel. Because of this, the multi-agent methods naturally have higher throughput capacity.

The Greedy Multi-Agent policy achieved the highest finished production count, with an average of about 72.84 finished filters, and also produced the highest total reward of about 963.44. This result is logical because the greedy policy directly follows the production structure: it produces missing components, presses when materials are available, and seals when semi-finished filters and bottom cloth are ready. Since the rules are already aligned with the manufacturing process, the greedy policy performs strongly in this controlled environment.

The Multi-Agent IQL model achieved an average of about 52.42 finished filters and a total reward of about 765.70. This is lower than the greedy policy, but still much stronger than the centralized DQN and random baseline. The result shows that Independent Q-Learning was able to learn useful cooperative behavior, even though each agent learns separately. However, the IQL method also produced the highest invalid actions, around 46.40 on average. This suggests that the agents learned to increase production activity, but coordination between agents was still imperfect. This is a common limitation of independent multi-agent learning because each agent learns from a changing environment created by the other learning agents.

The Centralized DQN produced the lowest invalid actions, around 10.20 on average, and maintained the lowest average WIP at about 4.87. This shows that the centralized agent behaved more conservatively and avoided many inefficient actions. However, it only produced about 15.97 finished filters on average and reached around 59.8% demand satisfaction. The lower throughput is mainly because the centralized agent can only execute one action per timestep, which limits its ability to operate multiple production stages in parallel.

The Random Multi-Agent baseline produced about 33.38 finished filters and reached relatively high demand satisfaction, but it also had the worst total reward at about -367.07. This happened because random agents frequently created invalid actions and excessive WIP. Even though some products were completed, the production behavior was inefficient and unstable. This confirms that high production alone is not enough; the scheduling policy must also control inventory buildup and avoid invalid operations.

Overall, the results show that parallel multi-agent control is more suitable for this type of manufacturing system because different production stages can operate at the same time. The Greedy Multi-Agent policy performed best because it used direct process knowledge, while Multi-Agent IQL showed that learning-based decentralized control can achieve strong performance but still suffers from coordination and credit assignment issues. The Centralized DQN was more stable and inventory-efficient, but its single-action control structure limited production capacity.