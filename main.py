import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH = 800
HEIGHT = 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Goku's Dragon Dodge")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

current_dir = os.path.dirname(os.path.abspath(__file__))
goku_img = pygame.image.load(os.path.join(current_dir, "pics/Goku.png"))
dragon_img = pygame.image.load(os.path.join(current_dir, "pics/Dragon.png"))

goku_img = pygame.transform.scale(goku_img, (60, 100))
dragon_img = pygame.transform.scale(dragon_img, (60, 100))

GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Goku:
    def __init__(self):
        self.rect = goku_img.get_rect()
        self.rect.centerx = WIDTH // 2
        self.rect.bottom = HEIGHT - 10
        self.speed = 5

    def move(self, action):
        if action == 0:  # Move left
            self.rect.x = max(0, self.rect.x - self.speed)
        elif action == 2:  # Move right
            self.rect.x = min(WIDTH - self.rect.width, self.rect.x + self.speed)
        # Action 1 is "do nothing"

class Dragon:
    def __init__(self):
        self.rect = dragon_img.get_rect()
        self.rect.x = random.randint(0, WIDTH - self.rect.width)
        self.rect.bottom = 0
        self.speed = random.randint(3, 7)

    def move(self):
        self.rect.y += self.speed

    def is_off_screen(self):
        return self.rect.top > HEIGHT

class Game:
    def __init__(self):
        self.goku = Goku()
        self.dragons = []
        self.score = 0
        self.high_score = 0
        self.frames = 0
        self.game_over = False

        # DQN setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_net = DQN(7, 3).to(self.device)
        self.target_net = DQN(7, 3).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=0.00025)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.epsilon = EPSILON_START

    def get_state(self):
        if not self.dragons:
            return np.array([0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])

        closest_left = min((dragon for dragon in self.dragons if dragon.rect.centerx < self.goku.rect.centerx),
                           key=lambda dragon: ((dragon.rect.centerx - self.goku.rect.centerx) ** 2 + (
                                       dragon.rect.centery - self.goku.rect.centery) ** 2),
                           default=None)
        closest_right = min((dragon for dragon in self.dragons if dragon.rect.centerx >= self.goku.rect.centerx),
                            key=lambda dragon: ((dragon.rect.centerx - self.goku.rect.centerx) ** 2 + (
                                        dragon.rect.centery - self.goku.rect.centery) ** 2),
                            default=None)
        closest_front = min(self.dragons, key=lambda
            dragon: dragon.rect.centery - self.goku.rect.centery if dragon.rect.centery > self.goku.rect.centery else float(
            'inf'))

        state = [
            self.goku.rect.centerx / WIDTH,  # Goku's x position
            min((dragon.rect.centery - self.goku.rect.centery) / HEIGHT for dragon in self.dragons),
            # Distance to closest dragon
            (closest_left.rect.centerx - self.goku.rect.centerx) / WIDTH if closest_left else 1.0,
            # Distance to closest left dragon
            (closest_right.rect.centerx - self.goku.rect.centerx) / WIDTH if closest_right else 1.0,
            # Distance to closest right dragon
            (closest_front.rect.centery - self.goku.rect.centery) / HEIGHT if closest_front.rect.centery > self.goku.rect.centery else 1.0,
            # Distance to closest front dragon
            min(abs(dragon.rect.centerx - self.goku.rect.centerx) / WIDTH for dragon in self.dragons),
            # Absolute horizontal distance to closest dragon
            len(self.dragons) / 10  # Number of dragons on screen (normalized)
        ]
        return np.array(state)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            state = self.get_state()

            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action = random.randint(0, 2)
            else:
                q_values = self.online_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                action = q_values.argmax().item()

            # Take action and observe reward
            self.goku.move(action)
            reward = self.update_game_state()

            # Store transition in replay buffer
            next_state = self.get_state()
            self.replay_buffer.append((state, action, reward, next_state, not self.game_over))

            # Train the network
            if len(self.replay_buffer) > MIN_REPLAY_SIZE:
                self.train()

            # Update target network
            if self.frames % TARGET_UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            # Decay epsilon
            self.epsilon = max(EPSILON_END,
                               EPSILON_START - (EPSILON_START - EPSILON_END) * (self.frames / EPSILON_DECAY))

            self.draw()
            clock.tick(60)
            self.frames += 1

            if self.game_over:
                self.reset_game()

        pygame.quit()

    def update_game_state(self):
        if random.random() < 0.01:  # 1% chance each frame to spawn a new dragon
            self.dragons.append(Dragon())

        reward = 0.1  # Small positive reward for surviving
        self.game_over = False

        for dragon in self.dragons:
            dragon.move()
            if dragon.is_off_screen():
                self.dragons.remove(dragon)
                reward += 0.5  # Additional reward for each dragon avoided
            elif self.goku.rect.colliderect(dragon.rect):
                reward = -1  # Negative reward for collision
                self.game_over = True
                break

        self.score += reward
        self.high_score = max(self.high_score, self.score)

        return reward

    def train(self):
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, not_dones = zip(*batch)

        # Convert lists to numpy arrays before creating tensors
        states = np.array(states)
        next_states = np.array(next_states)

        # Create tensors from numpy arrays
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        not_dones = torch.FloatTensor(not_dones).to(self.device)

        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + GAMMA * next_q_values * not_dones

        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def draw(self):
        window.fill(WHITE)
        window.blit(goku_img, self.goku.rect)
        for dragon in self.dragons:
            window.blit(dragon_img, dragon.rect)

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score:.2f}", True, GREEN)
        high_score_text = font.render(f"High Score: {self.high_score:.2f}", True, GREEN)
        frames_text = font.render(f"Frames: {self.frames}", True, GREEN)
        epsilon_text = font.render(f"Epsilon: {self.epsilon:.2f}", True, GREEN)
        window.blit(score_text, (10, 10))
        window.blit(high_score_text, (10, 50))
        window.blit(frames_text, (10, 90))
        window.blit(epsilon_text, (10, 130))

        pygame.display.flip()

    def reset_game(self):
        self.goku = Goku()
        self.dragons.clear()
        self.score = 0
        self.game_over = False

if __name__ == "__main__":
    game = Game()
    game.run()