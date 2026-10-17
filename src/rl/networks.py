"""Actor-Critic network architecture for discrete-action PPO.

:class:`ActorCritic` is a shared-trunk network: one MLP feature extractor
feeds two independent heads (Actor and Critic). This is the standard PPO
architecture (Schulman et al., 2017) -- sharing the trunk lets the
representation learned from the reward signal (via the critic) and the
policy gradient (via the actor) reinforce each other and halves the
parameter count versus two separate networks, at the cost of coupling
their optimization (mitigated here, as usual, by weighting the two loss
terms in the combined PPO loss -- see :mod:`scripts.train_ppo`).

Only :meth:`ActorCritic.forward` is intended to survive
``torch.jit.script`` export to production (see :func:`scripts.
train_ppo.export_model`): it is a pure, side-effect-free tensor-in/
tensor-out function, so the exported artifact's inference path never
touches the Python interpreter. :meth:`get_action_and_value` (which
builds a ``torch.distributions.Categorical``, a training-only
convenience) is deliberately *not* used by ``forward`` and is not part of
the scripted module's public surface as a result.
"""

from __future__ import annotations

import torch
from torch import nn


def masked_probabilities(probabilities: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Zero the forbidden actions' probabilities and renormalize.

    The network's actor head ends in a Softmax: it outputs PROBABILITIES, not logits, so masking by writing ``-inf`` (the logit
    convention) would be wrong here; the equivalent operation is to multiply by the mask and renormalize. If the policy put all its
    mass on forbidden actions (the sum is ~0), the result is uniform over the ALLOWED actions rather than NaN: a mask never leaves
    the agent without a legal move as long as at least one action is allowed.
    """
    allowed = action_mask.to(probabilities.dtype)
    masked = probabilities * allowed
    total = masked.sum(dim=-1, keepdim=True)
    uniform_over_allowed = allowed / allowed.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.where(total > 1e-12, masked / total.clamp_min(1e-12), uniform_over_allowed)


class ActorCritic(nn.Module):
    """Shared-MLP-trunk Actor-Critic network for a 3-action discrete policy.

    Attributes:
        shared: The common feature-extraction trunk both heads read from.
        actor_head: Outputs a categorical action distribution (via
            ``Softmax``) over the action space.
        critic_head: Outputs a single scalar state-value estimate.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        """Build the network.

        Args:
            observation_dim: Size of the flat observation vector (``6``
                for :class:`~src.rl.trading_env.MarketEnv`).
            action_dim: Number of discrete actions (``3``: HOLD/BUY/SELL).
            hidden_dim: Width of each shared hidden layer.
        """
        super().__init__()

        # Shared feature extractor. Tanh (not ReLU) is the standard choice
        # for PPO's small MLPs: it keeps activations bounded, which
        # empirically stabilizes the policy-gradient variance early in
        # training when the policy is still close to random.
        self.shared = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head: a categorical distribution over the 3 actions.
        # Softmax is applied here (rather than left as logits + an
        # external Categorical(logits=...)) because the spec calls for
        # the network itself to output action *probabilities*.
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic head: a single scalar V(s) estimate, plain Linear (no
        # activation -- a state-value can be any real number).
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the policy's action probabilities and the critic's value estimate.

        Args:
            observation: Shape ``(batch, observation_dim)`` (or
                ``(observation_dim,)`` for a single observation).

        Returns:
            ``(action_probabilities, state_value)``:
            ``action_probabilities`` has shape ``(batch, action_dim)`` and
            sums to ``1`` along the last dimension; ``state_value`` has
            shape ``(batch, 1)``.
        """
        features = self.shared(observation)
        action_probabilities = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_probabilities, state_value

    @torch.jit.unused
    def get_action_and_value(
        self, observation: torch.Tensor, action: torch.Tensor | None = None, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or re-evaluate) an action under the current policy.

        Used by both halves of the PPO loop in :mod:`scripts.train_ppo`:
        during rollout collection (``action=None``, freshly sample and
        record its log-probability) and during the update step (``action``
        given -- the action actually taken during rollout -- to
        re-evaluate its log-probability, entropy, and value estimate
        under the *current*, since-updated policy parameters).

        Decorated ``@torch.jit.unused``: this method builds a
        ``torch.distributions.Categorical``, which is training-only
        machinery with no place in the exported production artifact (see
        the module docstring).

        Args:
            observation: Shape ``(batch, observation_dim)``.
            action: If given, shape ``(batch,)`` of actions to evaluate
                instead of sampling fresh ones.
            action_mask: If given, boolean tensor of shape ``(batch, action_dim)`` (or ``(action_dim,)``):
                ``False`` = forbidden action. The policy's probabilities are zeroed on forbidden actions and
                renormalized, for sampling AND for the log-probability and entropy used by the PPO update (so the gradient
                is consistent with what was actually sampled). Never masks every action: see :func:`masked_probabilities`.

        Returns:
            ``(action, log_prob, entropy, value)``, each shape ``(batch,)``.
        """
        action_probabilities, state_value = self.forward(observation)
        if action_mask is not None:
            action_probabilities = masked_probabilities(action_probabilities, action_mask)
        distribution = torch.distributions.Categorical(action_probabilities)

        # torch.distributions.Categorical's methods are untyped in torch's
        # stubs (unlike the tensor ops around them); each call below is a
        # documented, stable part of the public Categorical API.
        if action is None:
            action = distribution.sample()  # type: ignore[no-untyped-call]

        log_prob = distribution.log_prob(action)  # type: ignore[no-untyped-call]
        entropy = distribution.entropy()  # type: ignore[no-untyped-call]
        return action, log_prob, entropy, state_value.squeeze(-1)
