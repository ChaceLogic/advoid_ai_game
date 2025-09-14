## Game Vars ##
NUM_CIRCLES = 5                           # Number of Circles model will take into count
CLOSEST_CIRCLE_REWARD_WEIGHT = 2          # the weight
EDGE_MARGIN = 50
EDGE_MARGIN_PUNISHMENT_WEIGHT = 0.5
WATCH_MODEL_TICK = 100
WATCH_MODEL_LEARN_FREQUENCY = 1000


## Training Vars ##
# MEMORY              : Replay buffer that stores the last 20,000 game experiences
# BATCH_SIZE          : Number of experiences used in each training step
# GAMMA               : Discount factor for future rewards     # Determines how much the AI values future vs immediate rewards
# EPSILON_START       : Initial exploration rate (100% random actions)
# EPSILON_END         : Minimum exploration rate (1% random actions)
# EPSILON_DECAY       : Rate at which exploration decreases
# TARGET_UPDATE       : Update target network every 10 episodes
# EPISODES            : Train for more episodes

# Base Model
# LEARNING_RATE = 0.0005
# MEMORY = 20000
# BATCH_SIZE = 128
# GAMMA = 0.99
# EPSILON_START = 1.0
# EPSILON_END = 0.01
# EPSILON_DECAY = 0.995
# TARGET_UPDATE = 10
# EPISODES = 500

# Advanced model
LEARNING_RATE = 1e-4      # Reduced for stability
MEMORY = 100000           # Larger replay buffer
BATCH_SIZE = 256          # Larger batch size
GAMMA = 0.99              # Keep unchanged
EPSILON_START = 1.0       # Keep unchanged
EPSILON_END = 0.05        # Slightly higher minimum exploration
EPSILON_DECAY = 0.995     # Keep unchanged
TARGET_UPDATE = 100       # Less frequent target updates
EPISODES = 1500           # Keep unchanged