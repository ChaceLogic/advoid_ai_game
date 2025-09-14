import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import random
from collections import deque
from vars import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Deep Q Network
# Learns constantly but makes mistakes
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# Target Network
# More stable network that updates less frequently and learns from DQN
class TargetNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc1 = nn.Linear(model.fc1.in_features, model.fc1.out_features)
        self.fc2 = nn.Linear(model.fc2.in_features, model.fc2.out_features)
        self.out = nn.Linear(model.out.in_features, model.out.out_features)
        self.hard_update(model)
        
    def hard_update(self, model):
        self.fc1.load_state_dict(model.fc1.state_dict())
        self.fc2.load_state_dict(model.fc2.state_dict())
        self.out.load_state_dict(model.out.state_dict())
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# Defines the square and circles
class GameObject():
    def __init__(self, x, y, speed, size):
        self.x = x
        self.y = y
        self.speed = speed
        self.size = size


# Game 
class AvoidGame:
    def __init__(self):
        self.WIDTH = 600       # Game Width
        self.HEIGHT = 400      # Game Height
        self.square_size = 30
        self.screen_diagonal_distance = math.sqrt(self.WIDTH**2 + self.HEIGHT**2)
        self.reset()

    def reset(self):
        # Centera aligns 
        self.square = GameObject(x=self.WIDTH // 2 - self.square_size // 2, 
                                y=self.HEIGHT - 50, 
                                speed=5, 
                                size=self.square_size)
        self.circles = []
        self.frame_count = 0
        self.spawn_delay = 50
        self.score = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        # Movement
        if action == 0:  # Left
            self.square.x -= self.square.speed
        # action == 1: # No Movement
        elif action == 2:  # Right
            self.square.x += self.square.speed

        # Keeps the square from going off-screen   
        self.square.x = max(0, min(self.WIDTH - self.square_size, self.square.x))

        # Spawn circles
        self.frame_count += 1
        if self.frame_count % self.spawn_delay == 0:
            circle = GameObject(
                x=random.randint(0, self.WIDTH - 30), 
                y=-15, 
                speed=3 + random.random() * 2,  # Vary speeds
                size=15
            )
            self.circles.append(circle)

        # Update circles
        for circle in self.circles:
            circle.y += circle.speed

        # Collision check
        for circle in self.circles:
            closest_x = max(self.square.x, min(circle.x, self.square.x + self.square.size))
            closest_y = max(self.square.y, min(circle.y, self.square.y + self.square.size))
            distance = math.sqrt((circle.x - closest_x)**2 + (circle.y - closest_y)**2)

            if distance < circle.size:
                self.done = True
                return self.get_state(), -10, True  # Reduced penalty

        # Reward components
        reward = 0.1  # Small survival bonus
        
        # Distance to closest circle
        if self.circles:
            # finds min_distance between all circles 
            min_distance = min(
                math.sqrt((c.x - self.square.x)**2 + (c.y - self.square.y)**2)    # a^2 + b^2 = c^2
                for c in self.circles
            )
            # Finds percentage of distance (0 = touching, 1 = max distance apart) and multiplys by weight
            reward += (min_distance / self.screen_diagonal_distance) * CLOSEST_CIRCLE_REWARD_WEIGHT

        # Edge avoidance (softer penalty)
        distance_from_edge = min(self.square.x, self.WIDTH - self.square.x)
        if distance_from_edge < EDGE_MARGIN:
            # Normalize and multiply by weight
            reward -= (1 - distance_from_edge/EDGE_MARGIN) * EDGE_MARGIN_PUNISHMENT_WEIGHT  # Much smaller penalty

        # Remove off-screen circles increase score
        self.score += sum(1 for c in self.circles if c.y >= self.HEIGHT)
        self.circles = [c for c in self.circles if c.y < self.HEIGHT]  
        
        return self.get_state(), reward, False

    def get_state(self):
        # Normalized state representation
        state = [
            self.square.x / self.WIDTH,  # 0-1
            self.square.y / self.HEIGHT   # Added y-coordinate
        ]

        # Sort circles by proximity to square (closest first)
        sorted_circles = sorted(
            self.circles,
            key=lambda c: math.sqrt((c.x-self.square.x)**2 + (c.y-self.square.y)**2)
        )

        for i in range(NUM_CIRCLES):
            if i < len(sorted_circles):
                circle = sorted_circles[i]
                state += [
                    circle.x / self.WIDTH,
                    circle.y / self.HEIGHT,
                    circle.speed / 5.0,  # Normalized speed
                    circle.size / 30.0  # Normalized size
                ]
            else:
                state += [0.0, 0.0, 0.0, 0.0]  # Padding
        # print(state)
        return np.array(state, dtype=np.float32)


def train_model():
    env = AvoidGame()
    input_dim = len(env.reset())
    model = DQN(input_dim, 3).to(device)
    target_net = TargetNetwork(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()

    # Training parameters
    memory = deque(maxlen=MEMORY)             # Replay buffer that stores the last 20,000 game experiences
    batch_size = BATCH_SIZE                   # Number of experiences used in each training step
    gamma = GAMMA                             # Discount factor for future rewards     # Determines how much the AI values future vs immediate rewards
    epsilon_start = EPSILON_START             # Initial exploration rate (100% random actions)
    epsilon_end = EPSILON_END                 # Minimum exploration rate (1% random actions)
    epsilon_decay = EPSILON_DECAY             # Rate at which exploration decreases
    target_update = TARGET_UPDATE             # Update target network every 10 episodes
    episodes = EPISODES                       # Train for more episodes
     
    epsilon = epsilon_start
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            # Descides weather to use a random action or a learned action
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            # Loads data into memory
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Train on batch
            if len(memory) >= batch_size:
                # Sample random batch from memory
                batch = random.sample(memory, batch_size)

                # Convert to tensors
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                # Current Q-values for taken actions
                current_q = model(states).gather(1, actions)

                # Target Q-values using target network
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + gamma * next_q * (1 - dones)   # Bellman equation

                # Calculate loss and update
                loss = loss_fn(current_q, target_q)     # Mean Squared Error (MSE)
                optimizer.zero_grad()                   # Clear old gradients
                loss.backward()                         # Compute new gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()                        # Update model weights

        # Update target network periodically
        if ep % target_update == 0:
            target_net.hard_update(model)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
        with open("train_log.txt", "a") as file:
            file.write(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}\n")

        # Occasionally render to see progress
        if ep % WATCH_MODEL_LEARN_FREQUENCY == 0:
            watch_ai(model, env, WATCH_MODEL_TICK, 'Training Visualization')

    return model


def watch_ai(model, env, tick, title, delay=0):
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(title)
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()

    state = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action = model(state_tensor).argmax().item()

        state, reward, done = env.step(action)

        # Draw everything
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, (0, 0, 255), (env.square.x, env.square.y, env.square.size, env.square.size))
        for circle in env.circles:
            pygame.draw.circle(screen, (255, 0, 0), (circle.x, circle.y), circle.size)
        score_text = font.render(f"Score: {env.score}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))
        pygame.display.update()
        clock.tick(tick)

        if delay > 0:
            time.sleep(delay)

    print(f'Score: {env.score}')
    pygame.quit()
    return env.score


def play_game(tick=60):
    env = AvoidGame()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption('Play Game')
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()

    state = env.reset()
    done = False

    while not done:
        action = 1  # Default: no movement
        
        # Handle keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0  # Move left
        elif keys[pygame.K_RIGHT]:
            action = 2  # Move right

        # Take game step with player's action
        state, reward, done = env.step(action)

        # Event handling (for closing window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Draw everything
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, (0, 0, 255), (env.square.x, env.square.y, env.square.size, env.square.size))
        for circle in env.circles:
            pygame.draw.circle(screen, (255, 0, 0), (circle.x, circle.y), circle.size)
        score_text = font.render(f"Score: {env.score}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))
        
        pygame.display.update()
        clock.tick(tick)
    
    print(f'Score: {env.score}')
    pygame.quit()
    return env.score


def main_menu():
    message = ""
    while True:
        print("\n" * 50)  # Clear screen effect
        print("==== MAIN MENU ====")
        print("1. Play the game")
        print("2. Train a model")
        print("3. Watch a model")
        print("4. Exit")
        print()

        if message:
            print(f"Message: {message}\n")
            message = ""  # Reset after displaying once

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            score = play_game()
            message = f'Score: {score}'
        elif choice == '2':
            train_model_name = input("Enter a filename to save the trained model:\n").strip()
            model = train_model()
            try:
                torch.save(model.state_dict(), f"{train_model_name}.pth")
                message = f"Model saved as '{train_model_name}.pth'"
            except Exception as e:
                message = f"Failed to save model: {e}"
        elif choice == '3':
            model_name = input("Enter the filename of the model to load:\n").strip()
            try:
                env = AvoidGame()
                input_dim = len(env.reset())
                model = DQN(input_dim=input_dim, output_dim=3).to(device)
                model.load_state_dict(torch.load(f"{model_name}.pth"))
                model.eval()
                score = watch_ai(model, env, WATCH_MODEL_TICK, 'Watch Model')
                message = f"Watching model '{model_name}'\n"
                message = message + f'Score: {score}'
            except Exception as e:
                message = f"Failed to load and watch model: {e}"
        elif choice == '4':
            print("Exiting...")
            break
        else:
            message = "Invalid choice. Please enter a number from 1 to 4."


main_menu()
