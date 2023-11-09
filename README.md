# Game Agent Training: Comparing SAC and PPO in Unity's 3DBall Environment

## Overview

This project aims to explore and evaluate the efficiency and effectiveness of two advanced reinforcement learning techniques: Soft Actor Critic (SAC) and Proximal Policy Optimization (PPO). By applying these methods to the challenging domain of game development, we seek to understand how these algorithms perform in a dynamic, continuous control task: the Unity 3DBall Balancing environment. This repository contains all the necessary code and resources for replicating our experiments, analyzing the performance of each algorithm, and exploring the nuances of reinforcement learning within game AI development.

## The Unity 3DBall Balancing Challenge

![Unity 3DBall Balancing](path/to/unity_3dball_image.png)
*The 3DBall Balancing Environment - a test of an agent's ability to balance a ball on a platform.*

The 3DBall Balancing game is a Unity environment where the goal is to balance a ball on a platform that an agent can tilt along two axes. It serves as an excellent benchmark for evaluating the adaptability and precision of RL algorithms in maintaining balance while responding to the continuous, unpredictable movement of the ball.

## Reinforcement Learning Algorithms

### Soft Actor Critic (SAC)

SAC is a state-of-the-art algorithm for deep reinforcement learning that employs a stochastic policy for efficient exploration of the action space. It is particularly noted for its stability and robustness, making it a prime candidate for environments with complex, high-dimensional spaces.

### Proximal Policy Optimization (PPO)

PPO has gained popularity due to its simplicity and effectiveness, especially in terms of sample efficiency and ease of tuning. It implements a trust region in policy optimization to prevent disruptive updates, ensuring smooth learning curves and consistent performance.

## Getting Started

### Integrating Unity with Python

We utilized the predefined example of the Unity 3DBall Balancing. By leveraging the ML-Agents toolkit provided by Unity Technologies, we constructed a compatible gym environment that can be interfaced using Python. This integration allows us to simulate the environment within a Python script, enabling the specific act of taking actions and retrieving observations directly from the Unity environment.

Here's an overview of the process we followed:

1. **Environment Setup:** We built the Unity environment using the ML-Agents toolkit, which involves setting up the Unity project and configuring the environment parameters.

2. **Gym Wrapper:** After building the Unity environment, we wrapped it with a gym interface. This custom wrapper translates the Unity environment's API into the OpenAI Gym interface, making it possible to use RL libraries that are compatible with Gym.

3. **Simulation and Interaction:** With the environment and gym wrapper in place, we can run simulations and interact with the environment using Python. This means we can implement, train, and evaluate RL algorithms like SAC and PPO in a Python script, directly influencing the Unity environment.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.6 or higher
- Unity ML-Agents
- Relevant Python libraries as specified in `requirements.txt`

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
