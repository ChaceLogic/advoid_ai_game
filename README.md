# Avoid Game with Deep Q-Learning

This project implements a simple avoidance game using **Pygame** and trains an AI agent to play it using **Deep Q-Learning (DQN)** with PyTorch.

The game consists of a **blue square (the player/agent)** that must avoid **red falling circles**.  
The AI learns to survive longer and score higher by avoiding collisions.

---

## Demo
Watch the advanced model below



## Project Setup Instructions

Follow these steps to set up and run the project on your local machine.

```bash
cd path/to/project-repository
mkdir env        # Create a directory for the virtual environment
cd env
python -m venv . # Create the virtual environment inside 'env'
cd ..
# On Windows
.\env\Scripts\activate
# On Linux
source env/bin/activate
pip install -r requirements.txt
python game.py
```

## Features

- **Game Mode**: Play the game manually using arrow keys.  
- **Train Mode**: Train a Deep Q-Network to play automatically.  
- **Watch Mode**: Load and watch a trained model play.  
- **Model Saving/Loading**: Save trained models to `.pth` files and reload them.  
- **Reinforcement Learning**: Uses a DQN with an experience replay buffer and target network for stable training.  

---

### Neural Networks

- **DQN**  
  - A Deep Q-Network that learns the best action to take based on the current state.  
  - **Input**: state representation (player + circles).  
  - **Output**: Q-values for 3 actions (**Left**, **Stay**, **Right**).  

- **TargetNetwork**  
  - A copy of the DQN that updates less frequently for stable learning.  
  - Uses **hard updates** from the main network every `TARGET_UPDATE` episodes.  

---

### Game Logic

- **GameObject**  
  - Represents a square (player) or circle (enemy).  
  - Stores position, speed, and size.  

- **AvoidGame**  
  - Defines the game world.  
  - `reset()` → Resets game state.  
  - `step(action)` → Applies an action, updates game state, returns `(state, reward, done)`.  
  - `get_state()` → Returns normalized state for the AI (square position + closest circles).  

---

### Training

- **train_model()**  
  - Runs Deep Q-Learning to train the agent.  
  - Uses **epsilon-greedy exploration**.  
  - Stores experiences in a **replay buffer** and trains on random batches.  
  - Updates **target network** periodically.  
  - Saves model when finished.  

---

## Rewards System

- **+0.1** → survival bonus each step.  
- **+distance-based reward** → for staying farther from circles.  
- **−penalty** → for being too close to screen edges.  
- **−10** → penalty for colliding with a circle (game over).  

---

## What is Fed into the Model  

The model receives a **state vector** that encodes:  

- **Player square position**:  
  - `square.x / WIDTH` (normalized 0–1)  
  - `square.y / HEIGHT` (normalized 0–1)  

- **Closest NUM_CIRCLES falling circles (sorted by proximity)**:  
  - `circle.x / WIDTH`  
  - `circle.y / HEIGHT`  
  - `circle.speed / 5.0` (normalized)  
  - `circle.size / 30.0` (normalized)  

*(If fewer than `NUM_CIRCLES`, padded with zeros.)*  

---

## Example Data  

```python
state = [
    0.32,   # Player x position (left side)
    0.88,   # Player y position
    
    # Circle 1 (closest)
    0.35,   # x position
    0.20,   # y position
    0.60,   # speed (3.0 / 5.0)
    0.50,   # size (15 / 30)
    
    # Circle 2
    0.70,   # x position
    0.10,   # y position
    0.70,   # speed (3.5 / 5.0)
    0.50,   # size
    
    # Circle 3 (padding)
    0.0, 0.0, 0.0, 0.0,
    # Circle 4
    0.0, 0.0, 0.0, 0.0,
    # Circle 5
    0.0, 0.0, 0.0, 0.0,
]
```