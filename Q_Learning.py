import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Check for GPU availability; if not, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingMLP(nn.Module):
    """
    Dueling Deep Q-Network (DQN) architecture.
    It splits the network into a Value head (state quality) and an
    Advantage head (action quality) to provide more stable Q-value estimates.
    """

    def __init__(self, input_dim=256):
        super(DuelingMLP, self).__init__()

        # Deep Feature Extraction network:
        # Designed to handle complex "multi-jump" logic and board formations.
        self.fe_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),  # Widened layer to store more complex formation memory
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1)
        )

        # Value Head: Estimates the scalar value of being in a specific state V(s)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage Head: Estimates the relative advantage of each action A(s, a)
        self.advantage_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Pass input through the shared feature extractor
        feat = self.fe_net(x)
        v = self.value_head(feat)
        a = self.advantage_head(feat)

        # Dueling formula: Q(s, a) = V(s) + (A(s, a) - Mean(A(s, a)))
        # This helps the model learn which states are fundamentally good or bad
        return v + (a - a.mean(dim=1, keepdim=True))


def encode_board(board, player_id):
    """
    Converts the 2D board list into a numerical tensor format for the NN.
    Uses 4 binary planes to represent: My Men, My Kings, Opponent Men, Opponent Kings.
    """
    opp_id = 3 - player_id
    planes = np.zeros((4, 8, 8), dtype=np.float32)

    for r in range(8):
        for c in range(8):
            cell = board[r][c]
            if cell == player_id:
                planes[0, r, c] = 1.0  # Current player's normal pieces
            elif cell == player_id + 2:
                planes[1, r, c] = 1.0  # Current player's kings
            elif cell == opp_id:
                planes[2, r, c] = 1.0  # Opponent's normal pieces
            elif cell == opp_id + 2:
                planes[3, r, c] = 1.0  # Opponent's kings

    # Perspective Normalization:
    # If the player is ID 2, we flip the board vertically.
    # This allows the AI to learn one set of "forward" strategies regardless of its side.
    if player_id == 2:
        planes = planes[:, ::-1, :].copy()

    # Flatten the (4, 8, 8) planes into a (256,) vector
    return torch.from_numpy(planes.flatten()).to(device)


class DQNAgent:
    def __init__(self, player_id=1):
        self.player_id = player_id

        # Used to decide actions and calculate gradients
        self.model = DuelingMLP().to(device)

        # Target Network: Used to provide stable Q-targets during updates (decreases oscillations)
        self.target_model = DuelingMLP().to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimizer: Using Adam with a low learning rate (1e-4) for stable fine-tuning
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        # Experience Replay Buffer: Stores past transitions to break correlation in training data
        self.memory = deque(maxlen=100000)

        self.batch_size = 128  # Increased Batch Size to stabilize learning of long-term logic
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1  # Initial exploration rate (100% random moves)

    def choose_action(self, board, moves):
        """
        Selects an action using the Epsilon-Greedy strategy.
        """
        # Exploration: Pick a random move
        if random.random() < self.epsilon:
            return random.choice(moves)

        # Exploitation: Pick the move that results in the highest predicted Q-value
        self.model.eval()
        best_val, best_move = -float('inf'), moves[0]

        with torch.no_grad():
            for move in moves:
                # Import update function locally to simulate the result of the move
                from Board_operations import update_board as _apply
                tmp = [row[:] for row in board]  # Create a deep copy of the board
                _apply(move[0], move[1], tmp)  # Simulate the move

                # Predict the value of the resulting board state
                val = self.model(encode_board(tmp, self.player_id).unsqueeze(0)).item()
                if val > best_val:
                    best_val, best_move = val, move

        self.model.train()  # Switch back to training mode
        return best_move

    def learn(self, s, r, sn, done):
        """
        Performs one step of Gradient Descent using sampled experiences.
        """
        # Add the transition to memory
        self.memory.append((s, r, sn, done))

        # Only start learning if we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a random batch of experiences (Replay Buffer)
        batch = random.sample(self.memory, self.batch_size)
        s_b, r_b, sn_b, d_b = zip(*batch)

        # Convert data to tensors and move to device
        s_b, sn_b = torch.stack(s_b), torch.stack(sn_b)
        r_b = torch.tensor(r_b, dtype=torch.float32).to(device)
        d_b = torch.tensor(d_b, dtype=torch.float32).to(device)

        # Get current Q-values from the Policy Network
        q_val = self.model(s_b).squeeze()

        # Calculate Target Q-values using the Target Network (Double DQN style logic)
        with torch.no_grad():
            next_q = self.target_model(sn_b).max(dim=1)[0] if self.target_model(sn_b).dim() > 1 else self.target_model(
                sn_b).squeeze()

        # Bellman Equation: Target = Reward + Gamma * Max(Next_Q) (if not terminal)
        target = r_b + self.gamma * next_q * (1.0 - d_b)

        # Loss function: SmoothL1 (Huber Loss) is less sensitive to outliers than MSE
        loss = nn.SmoothL1Loss()(q_val, target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    """Save network weights"""
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    """Load network weights"""
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))