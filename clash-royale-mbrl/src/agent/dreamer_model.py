"""
DreamerV3 World Model for Clash Royale.
Adapted from dreamer-pytorch for Apple Silicon MPS backend.

Key components:
- RSSM (Recurrent State Space Model) for world dynamics
- Encoder/Decoder for observations
- Actor-Critic for policy learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass

from ..utils.device import get_device, mps_safe_operation


@dataclass 
class DreamerConfig:
    """Configuration for DreamerV3 model."""
    # State dimensions
    stoch_size: int = 32        # Stochastic state size
    deter_size: int = 512       # Deterministic state size (GRU hidden)
    hidden_size: int = 512      # MLP hidden size
    
    # Observation space (grid tensor input)
    obs_channels: int = 8       # Number of grid channels
    obs_height: int = 18        # Grid height
    obs_width: int = 32         # Grid width
    
    # Action space
    num_cards: int = 5          # 0 = no action, 1-4 = cards
    grid_width: int = 32        # Deployment x positions
    grid_height: int = 18       # Deployment y positions
    
    # Training
    discount: float = 0.99
    lambda_: float = 0.95       # GAE lambda
    kl_free: float = 1.0        # Free nats for KL
    kl_scale: float = 1.0       # KL loss scale
    
    # Imagination
    horizon: int = 15           # Imagination rollout length


class RSSMState(NamedTuple):
    """State representation for RSSM."""
    stoch: torch.Tensor   # (batch, stoch_size) - sampled stochastic state
    deter: torch.Tensor   # (batch, deter_size) - deterministic state
    mean: torch.Tensor    # (batch, stoch_size) - distribution mean
    std: torch.Tensor     # (batch, stoch_size) - distribution std


def get_feat(state: RSSMState) -> torch.Tensor:
    """Concatenate stochastic and deterministic state."""
    return torch.cat([state.stoch, state.deter], dim=-1)


class GridEncoder(nn.Module):
    """
    Encodes grid tensor observations into latent embeddings.
    Uses conv layers suited for the 32x18 grid.
    """
    
    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        
        # Conv layers for grid encoding
        self.conv = nn.Sequential(
            nn.Conv2d(config.obs_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x9
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8x5 (approx)
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 4x3
            nn.ELU(),
            nn.Flatten(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, config.obs_channels, config.obs_height, config.obs_width)
            flat_size = self.conv(dummy).shape[1]
        
        self.fc = nn.Linear(flat_size, config.hidden_size)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, channels, height, width)
        Returns:
            embed: (batch, hidden_size)
        """
        x = self.conv(obs)
        return self.fc(x)


class GridDecoder(nn.Module):
    """
    Decodes latent state back to grid tensor (for reconstruction loss).
    """
    
    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        
        feat_size = config.stoch_size + config.deter_size
        
        # Project to initial feature map
        self.fc = nn.Linear(feat_size, 256 * 4 * 3)
        
        # Deconv layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ELU(),
            nn.Conv2d(32, config.obs_channels, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, state: RSSMState) -> torch.Tensor:
        """
        Args:
            state: RSSM state
        Returns:
            recon: (batch, channels, height, width)
        """
        feat = get_feat(state)
        x = self.fc(feat)
        x = x.view(-1, 256, 3, 4)
        x = self.deconv(x)
        
        # Resize to exact output size if needed
        if x.shape[-2:] != (self.config.obs_height, self.config.obs_width):
            x = F.interpolate(x, size=(self.config.obs_height, self.config.obs_width), mode='bilinear')
        
        return x


class RSSM(nn.Module):
    """
    Recurrent State Space Model.
    Models world dynamics with deterministic + stochastic components.
    """
    
    def __init__(self, config: DreamerConfig, action_size: int):
        super().__init__()
        self.config = config
        self.action_size = action_size
        
        # GRU for deterministic state
        self.gru = nn.GRUCell(config.hidden_size, config.deter_size)
        
        # Prior: predict next stochastic state from action + deterministic
        self.prior_fc = nn.Sequential(
            nn.Linear(action_size + config.stoch_size, config.hidden_size),
            nn.ELU(),
        )
        self.prior_out = nn.Linear(config.hidden_size, 2 * config.stoch_size)
        
        # Posterior: predict stochastic state from observation embed + deterministic
        self.posterior_fc = nn.Sequential(
            nn.Linear(config.hidden_size + config.deter_size, config.hidden_size),
            nn.ELU(),
        )
        self.posterior_out = nn.Linear(config.hidden_size, 2 * config.stoch_size)
    
    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Get initial RSSM state."""
        return RSSMState(
            stoch=torch.zeros(batch_size, self.config.stoch_size, device=device),
            deter=torch.zeros(batch_size, self.config.deter_size, device=device),
            mean=torch.zeros(batch_size, self.config.stoch_size, device=device),
            std=torch.ones(batch_size, self.config.stoch_size, device=device),
        )
    
    def _get_dist(self, mean: torch.Tensor, std: torch.Tensor) -> td.Distribution:
        """Create distribution from mean and std."""
        std = F.softplus(std) + 0.1  # Ensure positive std
        return td.Independent(td.Normal(mean, std), 1)
    
    def observe_step(self, prev_state: RSSMState, prev_action: torch.Tensor,
                     obs_embed: torch.Tensor) -> Tuple[RSSMState, RSSMState]:
        """
        Single observation step: compute prior and posterior.
        
        Args:
            prev_state: Previous RSSM state
            prev_action: Previous action (one-hot or embedding)
            obs_embed: Current observation embedding
            
        Returns:
            prior_state, posterior_state
        """
        # Compute prior
        prior_input = torch.cat([prev_action, prev_state.stoch], dim=-1)
        prior_hidden = self.prior_fc(prior_input)
        deter = self.gru(prior_hidden, prev_state.deter)
        
        prior_stats = self.prior_out(deter)
        prior_mean, prior_std_raw = prior_stats.chunk(2, dim=-1)
        prior_std = F.softplus(prior_std_raw) + 0.1
        
        # Sample prior
        prior_dist = self._get_dist(prior_mean, prior_std)
        prior_stoch = prior_dist.rsample()
        
        prior_state = RSSMState(
            stoch=prior_stoch, deter=deter,
            mean=prior_mean, std=prior_std
        )
        
        # Compute posterior
        posterior_input = torch.cat([obs_embed, deter], dim=-1)
        posterior_hidden = self.posterior_fc(posterior_input)
        posterior_stats = self.posterior_out(posterior_hidden)
        posterior_mean, posterior_std_raw = posterior_stats.chunk(2, dim=-1)
        posterior_std = F.softplus(posterior_std_raw) + 0.1
        
        # Sample posterior
        posterior_dist = self._get_dist(posterior_mean, posterior_std)
        posterior_stoch = posterior_dist.rsample()
        
        posterior_state = RSSMState(
            stoch=posterior_stoch, deter=deter,
            mean=posterior_mean, std=posterior_std
        )
        
        return prior_state, posterior_state
    
    def imagine_step(self, prev_state: RSSMState, action: torch.Tensor) -> RSSMState:
        """
        Imagination step: predict next state without observation.
        Used for policy learning via imagined rollouts.
        """
        prior_input = torch.cat([action, prev_state.stoch], dim=-1)
        prior_hidden = self.prior_fc(prior_input)
        deter = self.gru(prior_hidden, prev_state.deter)
        
        prior_stats = self.prior_out(deter)
        prior_mean, prior_std_raw = prior_stats.chunk(2, dim=-1)
        prior_std = F.softplus(prior_std_raw) + 0.1
        
        prior_dist = self._get_dist(prior_mean, prior_std)
        prior_stoch = prior_dist.rsample()
        
        return RSSMState(
            stoch=prior_stoch, deter=deter,
            mean=prior_mean, std=prior_std
        )


class RewardModel(nn.Module):
    """Predicts reward from RSSM state."""
    
    def __init__(self, config: DreamerConfig):
        super().__init__()
        feat_size = config.stoch_size + config.deter_size
        
        self.net = nn.Sequential(
            nn.Linear(feat_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, 1),
        )
    
    def forward(self, state: RSSMState) -> torch.Tensor:
        feat = get_feat(state)
        return self.net(feat).squeeze(-1)


class ContinueModel(nn.Module):
    """Predicts episode continuation (1 - done) from RSSM state."""
    
    def __init__(self, config: DreamerConfig):
        super().__init__()
        feat_size = config.stoch_size + config.deter_size
        
        self.net = nn.Sequential(
            nn.Linear(feat_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state: RSSMState) -> torch.Tensor:
        feat = get_feat(state)
        return self.net(feat).squeeze(-1)


class Actor(nn.Module):
    """
    Policy network for Clash Royale actions.
    Outputs: card selection (discrete) + deployment position (continuous grid)
    """
    
    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        feat_size = config.stoch_size + config.deter_size
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(feat_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
        )
        
        # Card selection head (discrete: 0=no action, 1-4=cards)
        self.card_head = nn.Linear(config.hidden_size, config.num_cards)
        
        # Position head (continuous x, y in [0, 1])
        self.pos_head = nn.Linear(config.hidden_size, 4)  # mean_x, mean_y, std_x, std_y
    
    def forward(self, state: RSSMState) -> Tuple[td.Distribution, td.Distribution]:
        """
        Returns:
            card_dist: Categorical distribution over card selection
            pos_dist: Normal distribution over deployment position
        """
        feat = get_feat(state)
        trunk_out = self.trunk(feat)
        
        # Card selection
        card_logits = self.card_head(trunk_out)
        card_dist = td.Categorical(logits=card_logits)
        
        # Position (only used if card != 0)
        pos_params = self.pos_head(trunk_out)
        pos_mean = torch.sigmoid(pos_params[..., :2])  # Keep in [0, 1]
        pos_std = F.softplus(pos_params[..., 2:]) + 0.01
        pos_dist = td.Independent(td.Normal(pos_mean, pos_std), 1)
        
        return card_dist, pos_dist
    
    def sample(self, state: RSSMState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and return log probability.
        
        Returns:
            card: (batch,) card index
            pos: (batch, 2) position [x, y]
            log_prob: (batch,) total log probability
        """
        card_dist, pos_dist = self.forward(state)
        
        card = card_dist.sample()
        pos = pos_dist.sample()
        
        log_prob = card_dist.log_prob(card) + pos_dist.log_prob(pos)
        
        return card, pos, log_prob


class Critic(nn.Module):
    """Value network for RSSM states."""
    
    def __init__(self, config: DreamerConfig):
        super().__init__()
        feat_size = config.stoch_size + config.deter_size
        
        self.net = nn.Sequential(
            nn.Linear(feat_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, 1),
        )
    
    def forward(self, state: RSSMState) -> torch.Tensor:
        feat = get_feat(state)
        return self.net(feat).squeeze(-1)


class DreamerV3Model(nn.Module):
    """
    Complete DreamerV3 model combining all components.
    """
    
    def __init__(self, config: Optional[DreamerConfig] = None):
        super().__init__()
        self.config = config or DreamerConfig()
        
        # Calculate action embedding size
        action_size = self.config.num_cards + 2  # one-hot card + position
        
        # Components
        self.encoder = GridEncoder(self.config)
        self.decoder = GridDecoder(self.config)
        self.rssm = RSSM(self.config, action_size)
        self.reward_model = RewardModel(self.config)
        self.continue_model = ContinueModel(self.config)
        self.actor = Actor(self.config)
        self.critic = Critic(self.config)
    
    def encode_action(self, card: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Encode action as single tensor.
        
        Args:
            card: (batch,) card index
            pos: (batch, 2) position
            
        Returns:
            action_embed: (batch, action_size)
        """
        # One-hot encode card
        card_onehot = F.one_hot(card.long(), self.config.num_cards).float()
        return torch.cat([card_onehot, pos], dim=-1)
    
    def to_device(self, device: torch.device = None):
        """Move model to device."""
        if device is None:
            device = get_device()
        return self.to(device)


if __name__ == "__main__":
    # Test model instantiation
    print("Testing DreamerV3 Model...")
    
    device = get_device()
    print(f"Using device: {device}")
    
    config = DreamerConfig()
    model = DreamerV3Model(config).to(device)
    
    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, config.obs_channels, config.obs_height, config.obs_width, device=device)
    
    # Encode observation
    embed = model.encoder(obs)
    print(f"Encoder output shape: {embed.shape}")
    
    # Initialize state
    state = model.rssm.initial_state(batch_size, device)
    print(f"Initial state - stoch: {state.stoch.shape}, deter: {state.deter.shape}")
    
    # Test action sampling
    card, pos, log_prob = model.actor.sample(state)
    print(f"Sampled action - card: {card.shape}, pos: {pos.shape}")
    
    # Encode action
    action = model.encode_action(card, pos)
    print(f"Encoded action: {action.shape}")
    
    # Test RSSM step
    prior, posterior = model.rssm.observe_step(state, action, embed)
    print(f"RSSM step complete - posterior stoch: {posterior.stoch.shape}")
    
    # Test critic
    value = model.critic(posterior)
    print(f"Critic value: {value.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
