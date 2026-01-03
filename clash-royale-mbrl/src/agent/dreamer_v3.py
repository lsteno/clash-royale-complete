"""
DreamerV3 World Model adapted from dreamer-pytorch.

This implements the core RSSM (Recurrent State Space Model) that makes
Dreamer different from simple model-free RL:

1. RSSM learns to predict future states (world model)
2. Agent can "imagine" trajectories without real gameplay
3. Actor/Critic are trained on imagined trajectories

Adapted for:
- PyTorch with MPS (Apple Silicon) support
- Clash Royale state space
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class RSSMState:
    """State of the RSSM world model."""
    mean: torch.Tensor      # Mean of stochastic state distribution
    std: torch.Tensor       # Std of stochastic state distribution  
    stoch: torch.Tensor     # Sampled stochastic state
    deter: torch.Tensor     # Deterministic state (GRU hidden)
    
    def detach(self) -> 'RSSMState':
        return RSSMState(
            mean=self.mean.detach(),
            std=self.std.detach(),
            stoch=self.stoch.detach(),
            deter=self.deter.detach()
        )
    
    @property
    def features(self) -> torch.Tensor:
        """Concatenate stochastic and deterministic for downstream use."""
        return torch.cat([self.stoch, self.deter], dim=-1)
    
    @property
    def distribution(self) -> td.Distribution:
        """Get the distribution for KL divergence."""
        return td.Independent(td.Normal(self.mean, self.std), 1)


class ObservationEncoder(nn.Module):
    """
    CNN encoder for visual observations.
    Input: (B, 3, 256, 128) - arena image
    Output: (B, embed_size) - embedding vector
    """
    def __init__(self, shape: Tuple[int, int, int] = (3, 256, 128), 
                 depth: int = 32, embed_size: int = 256):
        super().__init__()
        self.shape = shape
        c, h, w = shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, depth, 4, stride=2, padding=1),      # 128x64
            nn.ReLU(),
            nn.Conv2d(depth, 2*depth, 4, stride=2, padding=1),   # 64x32
            nn.ReLU(),
            nn.Conv2d(2*depth, 4*depth, 4, stride=2, padding=1), # 32x16
            nn.ReLU(),
            nn.Conv2d(4*depth, 8*depth, 4, stride=2, padding=1), # 16x8
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy).shape[1]
        
        self.fc = nn.Linear(conv_out, embed_size)
        self.embed_size = embed_size
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, C, H, W) or (B, T, C, H, W)
        Returns:
            embed: (B, embed_size) or (B, T, embed_size)
        """
        has_time = obs.dim() == 5
        if has_time:
            B, T = obs.shape[:2]
            obs = obs.reshape(B * T, *obs.shape[2:])
        
        x = self.conv(obs)
        embed = self.fc(x)
        
        if has_time:
            embed = embed.reshape(B, T, -1)
        
        return embed


class ObservationDecoder(nn.Module):
    """
    CNN decoder for reconstructing observations.
    Used to ensure the world model captures visual information.
    """
    def __init__(self, feature_size: int, shape: Tuple[int, int, int] = (3, 256, 128),
                 depth: int = 32):
        super().__init__()
        self.shape = shape
        c, h, w = shape
        
        # Starting shape after deconv
        self.start_h = h // 16
        self.start_w = w // 16
        self.start_c = 8 * depth
        
        self.fc = nn.Linear(feature_size, self.start_c * self.start_h * self.start_w)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8*depth, 4*depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4*depth, 2*depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*depth, depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(depth, c, 4, stride=2, padding=1),
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_size) or (B, T, feature_size)
        Returns:
            reconstruction: (B, C, H, W) or (B, T, C, H, W)
        """
        has_time = features.dim() == 3
        if has_time:
            B, T = features.shape[:2]
            features = features.reshape(B * T, -1)
        
        x = self.fc(features)
        x = x.reshape(-1, self.start_c, self.start_h, self.start_w)
        recon = self.deconv(x)
        
        if has_time:
            recon = recon.reshape(B, T, *recon.shape[1:])
        
        return recon


class RSSMTransition(nn.Module):
    """
    RSSM Transition (Prior) model.
    Predicts next state given previous state and action.
    This is the "imagination" part - predicts without observation.
    """
    def __init__(self, action_size: int, stoch_size: int = 32, 
                 deter_size: int = 256, hidden_size: int = 256):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        
        # Input: previous action + previous stochastic state
        self.rnn_input = nn.Sequential(
            nn.Linear(action_size + stoch_size, hidden_size),
            nn.ELU(),
        )
        
        # GRU for deterministic state
        self.gru = nn.GRUCell(hidden_size, deter_size)
        
        # Output: parameters for stochastic state distribution
        self.stoch_model = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size),  # mean and std
        )
        
    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Create initial zero state."""
        return RSSMState(
            mean=torch.zeros(batch_size, self.stoch_size, device=device),
            std=torch.ones(batch_size, self.stoch_size, device=device),
            stoch=torch.zeros(batch_size, self.stoch_size, device=device),
            deter=torch.zeros(batch_size, self.deter_size, device=device),
        )
    
    def forward(self, prev_action: torch.Tensor, prev_state: RSSMState) -> RSSMState:
        """
        Predict next state (prior) without observation.
        
        Args:
            prev_action: (B, action_size)
            prev_state: Previous RSSM state
        Returns:
            next_state: Predicted next state
        """
        # Combine action and previous stochastic state
        x = torch.cat([prev_action, prev_state.stoch], dim=-1)
        x = self.rnn_input(x)
        
        # Update deterministic state
        deter = self.gru(x, prev_state.deter)
        
        # Predict stochastic state distribution
        stats = self.stoch_model(deter)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + 0.1  # Ensure positive std
        
        # Sample stochastic state
        dist = td.Normal(mean, std)
        stoch = dist.rsample()
        
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)


class RSSMRepresentation(nn.Module):
    """
    RSSM Representation (Posterior) model.
    Updates state given observation embedding.
    This is the "grounding" part - corrects imagination with reality.
    """
    def __init__(self, embed_size: int, stoch_size: int = 32, 
                 deter_size: int = 256, hidden_size: int = 256):
        super().__init__()
        self.stoch_size = stoch_size
        
        # Combine observation embedding with deterministic state
        self.posterior = nn.Sequential(
            nn.Linear(embed_size + deter_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size),
        )
    
    def forward(self, obs_embed: torch.Tensor, prior_state: RSSMState) -> RSSMState:
        """
        Update state with observation (posterior).
        
        Args:
            obs_embed: (B, embed_size) observation embedding
            prior_state: State from transition model
        Returns:
            posterior_state: Updated state with observation
        """
        # Combine observation with deterministic state
        x = torch.cat([obs_embed, prior_state.deter], dim=-1)
        stats = self.posterior(x)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + 0.1
        
        # Sample from posterior
        dist = td.Normal(mean, std)
        stoch = dist.rsample()
        
        return RSSMState(
            mean=mean, std=std, stoch=stoch, 
            deter=prior_state.deter  # Keep deterministic from prior
        )


class RewardPredictor(nn.Module):
    """Predict reward from RSSM state features."""
    def __init__(self, feature_size: int, hidden_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


class ContinuePredictor(nn.Module):
    """Predict episode continuation (1 - done) from features."""
    def __init__(self, feature_size: int, hidden_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(features)).squeeze(-1)


class Actor(nn.Module):
    """
    Actor network for Clash Royale.
    Outputs: card selection (0-4) and position (0-8)
    """
    def __init__(self, feature_size: int, n_cards: int = 5, n_positions: int = 9,
                 hidden_size: int = 256):
        super().__init__()
        self.n_cards = n_cards
        self.n_positions = n_positions
        
        self.shared = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        
        self.card_head = nn.Linear(hidden_size, n_cards)
        self.pos_head = nn.Linear(hidden_size, n_positions)
    
    def forward(self, features: torch.Tensor, 
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, feature_size) RSSM state features
            deterministic: If True, return argmax instead of sample
        Returns:
            card: (B,) selected card index
            pos: (B,) selected position index
        """
        h = self.shared(features)
        
        card_logits = self.card_head(h)
        pos_logits = self.pos_head(h)
        
        if deterministic:
            card = card_logits.argmax(dim=-1)
            pos = pos_logits.argmax(dim=-1)
        else:
            card = td.Categorical(logits=card_logits).sample()
            pos = td.Categorical(logits=pos_logits).sample()
        
        return card, pos
    
    def log_prob(self, features: torch.Tensor, card: torch.Tensor, 
                 pos: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions."""
        h = self.shared(features)
        
        card_logits = self.card_head(h)
        pos_logits = self.pos_head(h)
        
        card_log_prob = td.Categorical(logits=card_logits).log_prob(card)
        pos_log_prob = td.Categorical(logits=pos_logits).log_prob(pos)
        
        return card_log_prob + pos_log_prob


class Critic(nn.Module):
    """Value function for estimating expected returns."""
    def __init__(self, feature_size: int, hidden_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


class DreamerV3(nn.Module):
    """
    Complete DreamerV3 world model and policy.
    
    Components:
    1. Encoder: obs -> embedding
    2. RSSM: embedding -> latent state (stoch + deter)
    3. Decoder: state -> reconstructed obs
    4. Reward model: state -> predicted reward
    5. Actor: state -> action
    6. Critic: state -> value
    """
    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (3, 256, 128),
        action_size: int = 45,  # 5 cards * 9 positions
        n_cards: int = 5,
        n_positions: int = 9,
        embed_size: int = 256,
        stoch_size: int = 32,
        deter_size: int = 256,
        hidden_size: int = 256,
    ):
        super().__init__()
        
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.feature_size = stoch_size + deter_size
        
        # World model components
        self.encoder = ObservationEncoder(obs_shape, embed_size=embed_size)
        self.decoder = ObservationDecoder(self.feature_size, obs_shape)
        self.transition = RSSMTransition(action_size, stoch_size, deter_size, hidden_size)
        self.representation = RSSMRepresentation(embed_size, stoch_size, deter_size, hidden_size)
        self.reward_predictor = RewardPredictor(self.feature_size, hidden_size)
        self.continue_predictor = ContinuePredictor(self.feature_size, hidden_size)
        
        # Policy components
        self.actor = Actor(self.feature_size, n_cards, n_positions, hidden_size)
        self.critic = Critic(self.feature_size, hidden_size)
        
    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Get initial RSSM state."""
        return self.transition.initial_state(batch_size, device)
    
    def observe_step(self, obs: torch.Tensor, action: torch.Tensor, 
                     state: RSSMState) -> Tuple[RSSMState, RSSMState]:
        """
        Single step of observation.
        
        Args:
            obs: (B, C, H, W) observation
            action: (B, action_size) one-hot action
            state: Previous RSSM state
            
        Returns:
            prior: State predicted by transition (imagination)
            posterior: State updated with observation (reality)
        """
        # Predict next state (prior/imagination)
        prior = self.transition(action, state)
        
        # Encode observation
        obs_embed = self.encoder(obs)
        
        # Update with observation (posterior/reality)
        posterior = self.representation(obs_embed, prior)
        
        return prior, posterior
    
    def imagine_step(self, action: torch.Tensor, state: RSSMState) -> RSSMState:
        """
        Single step of imagination (no observation).
        Used for training actor/critic in imagined trajectories.
        """
        return self.transition(action, state)
    
    def imagine_trajectory(self, initial_state: RSSMState, 
                          horizon: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imagine a trajectory using the actor policy.
        
        Args:
            initial_state: Starting RSSM state
            horizon: Number of steps to imagine
            
        Returns:
            features: (B, T, feature_size) imagined state features
            rewards: (B, T) predicted rewards
            continues: (B, T) predicted continuation
        """
        features_list = []
        rewards_list = []
        continues_list = []
        
        state = initial_state
        batch_size = state.stoch.shape[0]
        device = state.stoch.device
        
        for _ in range(horizon):
            # Get action from actor
            card, pos = self.actor(state.features, deterministic=False)
            
            # Convert to one-hot action
            action = torch.zeros(batch_size, 5 * 9, device=device)
            action_idx = card * 9 + pos
            action.scatter_(1, action_idx.unsqueeze(1), 1)
            
            # Imagine next state
            state = self.imagine_step(action, state)
            
            # Predict reward and continuation
            features_list.append(state.features)
            rewards_list.append(self.reward_predictor(state.features))
            continues_list.append(self.continue_predictor(state.features))
        
        features = torch.stack(features_list, dim=1)
        rewards = torch.stack(rewards_list, dim=1)
        continues = torch.stack(continues_list, dim=1)
        
        return features, rewards, continues


def create_dreamer_model(device: torch.device) -> DreamerV3:
    """Create and initialize DreamerV3 model."""
    model = DreamerV3(
        obs_shape=(3, 256, 128),
        n_cards=5,
        n_positions=9,
        embed_size=256,
        stoch_size=32,
        deter_size=256,
        hidden_size=256,
    ).to(device)
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_dreamer_model(device)
    
    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, 3, 256, 128, device=device)
    action = torch.zeros(batch_size, 45, device=device)
    action[:, 0] = 1  # No action
    
    state = model.initial_state(batch_size, device)
    prior, posterior = model.observe_step(obs, action, state)
    
    print(f"Prior stoch: {prior.stoch.shape}")
    print(f"Posterior stoch: {posterior.stoch.shape}")
    print(f"Features: {posterior.features.shape}")
    
    # Test imagination
    features, rewards, continues = model.imagine_trajectory(posterior, horizon=15)
    print(f"Imagined features: {features.shape}")
    print(f"Imagined rewards: {rewards.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
