# Q-Learning Agent for Grid World Environment

This project implements a Q-learning agent that learns to navigate a grid world environment using reinforcement learning. The agent receives rewards for reaching the goal, penalties for hitting obstacles, and adaptive penalties for long episodes and loops.

---

## **Project Structure**

- `q_learning_agent.py` — The main script that trains the Q-learning agent.
- `environment.py` — Defines the grid world environment and reward structure.
- `requirements.txt` — Contains all the dependencies needed to run the project.

---

## **Setup Instructions (use the git bash CLI)**

### 1. **Clone the Repository**

First, clone the repository to your local machine:
```bash
git clone https://github.com/enepac/q-learning.git

cd q-learning
```
On Windows(use the git bash CLI):
```bash
python -m venv venv
source venv/Scripts/activate
```

### Install Dependencies
Install all required packages from requirements.txt:
```bash
pip install -r requirements.txt
```
### Run the Project
Once the setup is complete, run the Q-learning agent:
```bash
python q_learning_agent.py
```

