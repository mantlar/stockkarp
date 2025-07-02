import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import logging

# Initialize logger
logging.basicConfig(
    filename=f"training-{time.ctime(time.time())}.log".replace(" ", "").replace(":", ""),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class DQN(nn.Module):
    """Deep Q-Network model"""

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Learning Agent"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.sync_target_model()

    def sync_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        # Ensure state and next_state are tensors
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        # print(f"Remembering {state}, {action}, {reward}, {next_state}, {done}")
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions_mask):
        """Choose action using epsilon-greedy policy"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        if np.random.rand() <= self.epsilon:
            # Random action from valid choices
            logging.info("Choosing random option")
            valid_indices = [i for i, valid in enumerate(
                valid_actions_mask) if valid]
            if not valid_indices:
                return 0  # Fallback action if no valid actions
            return random.choice(valid_indices)
        else:
            logging.info("Using model to make decision")
            with torch.no_grad():
                q_values = self.model(state.unsqueeze(0))
            q_values = q_values.squeeze(0).numpy()

            # Apply valid actions mask
            q_values = q_values * valid_actions_mask
            q_values[valid_actions_mask == 0] = -np.inf

            return np.argmax(q_values)

    def replay(self, batch_size):
        """Train on a batch from replay memory"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        logging.info("Replaying %s experiences from memory", batch_size)

        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)

        # Get current Q values
        current_q = self.model(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)
        logging.info("Current Q values: %s", current_q)

        # Get target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].detach()
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        logging.info("Target Q values: %s", target_q)

        # Compute loss
        loss = self.criterion(current_q, target_q)
        logging.info("Loss before optimization: %s", loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logging.info("Current epsilon value: %s", self.epsilon)

        # Sync target model periodically
        if random.random() < 0.01:
            self.sync_target_model()
            logging.info("Target model synchronized")

        logging.info("Loss after optimization: %s", loss.item())
        return loss.item()

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.sync_target_model()


torch.serialization.add_safe_globals([DQN])
torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear])
