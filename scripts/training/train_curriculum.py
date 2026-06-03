"""
Curriculum learning : entraîne le LEAGUE_CHAMPION par étapes
1H+2R → 2H+1R → 3H, avec éval rigoureuse entre chaque phase.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from tianshou.env import DummyVectorEnv
    from tianshou.policy import PPOPolicy
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.trainer import onpolicy_trainer
except ModuleNotFoundError:
    DummyVectorEnv = None
    PPOPolicy = None
    Collector = None
    VectorReplayBuffer = None
    onpolicy_trainer = None

from love_letter.engine import LoveLetterRLEnv
from love_letter.bots.heuristic import HeuristicBot
from love_letter.paths import checkpoint_path

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128
NUM_OPPONENTS = 3

STARTING_CHECKPOINT = checkpoint_path("ppo_love_letter_league_champion.pth")


def _require_tianshou():
    if PPOPolicy is None:
        raise RuntimeError(
            "Les APIs Tianshou attendues par les scripts d'entraînement ne sont pas "
            "disponibles dans cet environnement. L'évaluation pure reste utilisable."
        )


# ==========================================
# WRAPPER PARAMÉTRABLE (n_heuristics)
# ==========================================
class CurriculumWrapper(gym.Env):
    """
    Wrapper qui place player_0 face à n_heuristics adversaires heuristiques
    et (3 - n_heuristics) adversaires random.

    Les heuristics sont placées sur les premiers slots adverses (player_1, player_2...)
    pour que la position soit déterministe.
    """
    def __init__(self, pz_env, agent_id="player_0", n_heuristics=1):
        super().__init__()
        self.env = pz_env
        self.agent_id = agent_id
        self.n_heuristics = n_heuristics
        self.observation_space = self.env.observation_space(agent_id)
        self.action_space = self.env.action_space(agent_id)
        self.teacher = HeuristicBot()

        # Liste des seats adverses : ceux avec heuristique en premier
        self.heuristic_seats = set(
            f"player_{i}" for i in range(1, 1 + n_heuristics)
        )

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        obs, reward, term, trunc, info = self._step_until_my_turn()
        while term or trunc:
            self.env.reset(seed=seed, options=options)
            obs, reward, term, trunc, info = self._step_until_my_turn()
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.last()
        if term or trunc:
            self.env.step(None)
        else:
            self.env.step(action)
        return self._step_until_my_turn()

    def _step_until_my_turn(self):
        while True:
            if getattr(self.env, "agent_selection", None) is None:
                return self.observation_space.sample(), 0.0, True, False, {}

            agent = self.env.agent_selection
            obs, reward, term, trunc, info = self.env.last()

            if agent == self.agent_id:
                return obs, reward, term, trunc, info

            if term or trunc:
                self.env.step(None)
            else:
                if agent in self.heuristic_seats:
                    act = self.teacher.choose_action(self.env, agent)
                else:
                    mask = obs["action_mask"]
                    valid = np.where(mask == 1)[0]
                    act = int(np.random.choice(valid)) if len(valid) else 0
                self.env.step(act)


# ==========================================
# RÉSEAU
# ==========================================
class RecurrentEncoder(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_size=LATENT):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, hidden_size),
            nn.ReLU()
        )
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, obs, state=None, info={}):
        if hasattr(obs, 'observation'):
            x_val = obs.observation
        elif isinstance(obs, dict):
            x_val = obs['observation']
        else:
            x_val = obs

        x = torch.as_tensor(x_val, dtype=torch.float32, device=next(self.parameters()).device)
        features = self.feature_extractor(x)

        if state is None or (hasattr(state, 'is_empty') and state.is_empty()):
            h_in = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        else:
            if isinstance(state, dict) and 'hidden' in state:
                h_in = state['hidden']
            elif hasattr(state, 'hidden'):
                h_in = state.hidden
            else:
                h_in = state
            h_in = torch.as_tensor(h_in, dtype=torch.float32, device=x.device)

        if h_in.shape[0] != x.shape[0]:
            h_in = torch.zeros(x.shape[0], self.hidden_size, device=x.device)

        new_state = self.rnn(features, h_in)
        return new_state, {"hidden": new_state.detach().cpu().numpy()}


class MaskedActor(nn.Module):
    def __init__(self, encoder, action_dim=ACTION_DIM, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, action_dim)

    def forward(self, obs, state=None, info={}):
        features, state = self.encoder(obs, state, info)
        logits = self.head(features)
        if hasattr(obs, 'action_mask'):
            mask_val = obs.action_mask
        elif isinstance(obs, dict):
            mask_val = obs['action_mask']
        else:
            mask_val = obs
        mask = torch.as_tensor(mask_val, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, state


class Critic(nn.Module):
    def __init__(self, encoder, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, 1)

    def forward(self, obs, state=None, info={}):
        features, _ = self.encoder(obs, state, info)
        return self.head(features).flatten()


class BeliefHead(nn.Module):
    def __init__(self, encoder, num_opponents=NUM_OPPONENTS, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.num_opponents = num_opponents
        self.head = nn.Linear(latent, num_opponents * 10)

    def forward(self, obs, state=None, info={}):
        features, _ = self.encoder(obs, state, info)
        logits = self.head(features)
        return logits.view(-1, self.num_opponents, 10)


if PPOPolicy is not None:
    class AuxPPOPolicy(PPOPolicy):
        def __init__(self, belief_head, belief_coef=0.05, **kwargs):
            super().__init__(**kwargs)
            self.belief_head = belief_head
            self.belief_coef = belief_coef
            self.belief_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        def learn(self, batch, batch_size, repeat, **kwargs):
            res = super().learn(batch, batch_size, repeat, **kwargs)

            device = next(self.belief_head.parameters()).device
            state = batch.state if hasattr(batch, 'state') else None

            belief_logits = self.belief_head(batch.obs, state)
            hidden_cards = torch.as_tensor(batch.info.hidden_cards, dtype=torch.long, device=device)

            B = belief_logits.shape[0]
            loss_belief = self.belief_loss_fn(
                belief_logits.reshape(B * self.belief_head.num_opponents, 10),
                hidden_cards.reshape(B * self.belief_head.num_opponents),
            )

            self.optim.zero_grad()
            (self.belief_coef * loss_belief).backward()
            self.optim.step()

            res["loss/belief"] = loss_belief.item()
            return res
else:
    class AuxPPOPolicy:
        def __init__(self, *args, **kwargs):
            _require_tianshou()


# ==========================================
# ÉVALUATION RIGOUREUSE
# ==========================================
def evaluate_actor(actor, n_heuristics_list=[0, 1, 2, 3], n_games=500):
    """
    Évalue le modèle dans plusieurs configs et renvoie un dict {config: stats}.
    Utilise la méthode propre de evaluate_propre.py.
    """
    bot = HeuristicBot()
    results = {}

    for n_h in n_heuristics_list:
        env_eval = LoveLetterRLEnv(num_players=4)
        rewards = np.zeros(n_games)

        for game in range(n_games):
            env_eval.reset(seed=10_000 + game)
            state = None

            for agent in env_eval.agent_iter():
                obs_dict, r, term, trunc, _ = env_eval.last()
                if agent == "player_0":
                    rewards[game] += r
                if term or trunc:
                    env_eval.step(None)
                    continue

                if agent == "player_0":
                    with torch.no_grad():
                        x = torch.as_tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
                        mask = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool).unsqueeze(0)
                        h_in = state if state is not None else torch.zeros(1, LATENT)
                        feat, new_state = actor.encoder.feature_extractor(x), None
                        # Utilise l'API encoder normale
                        new_state_full = actor.encoder.rnn(actor.encoder.feature_extractor(x), h_in)
                        logits = actor.head(new_state_full)
                        logits = logits.masked_fill(~mask, -1e9)
                        action = int(logits.argmax(dim=-1).item())
                        state = new_state_full
                else:
                    seat_idx = int(agent[-1])
                    is_heuristic = seat_idx in range(1, 1 + n_h)
                    if is_heuristic:
                        action = bot.choose_action(env_eval, agent)
                    else:
                        mask = obs_dict["action_mask"]
                        valid = np.where(mask == 1)[0]
                        action = int(np.random.choice(valid)) if len(valid) else 0

                env_eval.step(action)

        mean_r = float(rewards.mean())
        winrate = float((rewards >= 1.0).mean())
        ci = 1.96 * float(rewards.std()) / np.sqrt(n_games)
        results[f"vs_{n_h}H"] = {
            "reward": mean_r,
            "ci_95": ci,
            "winrate": winrate,
        }

    return results


def print_eval(label, results):
    print(f"\n{'─' * 60}")
    print(f"  ÉVALUATION : {label}")
    print(f"{'─' * 60}")
    for cfg, r in results.items():
        print(f"  {cfg:8s} | reward={r['reward']:.3f} ± {r['ci_95']:.3f}  "
              f"winrate={r['winrate']:.3f}")


# ==========================================
# UNE PHASE D'ENTRAÎNEMENT
# ==========================================
def train_phase(actor, critic, encoder, belief_head, n_heuristics, n_epochs, label, lr=5e-5):
    """Entraîne sur une config spécifique."""
    _require_tianshou()
    print(f"\n{'='*60}")
    print(f"  PHASE : {label}")
    print(f"  Config : 1 PPO vs {n_heuristics}H + {3 - n_heuristics}R")
    print(f"  Epochs : {n_epochs} | LR : {lr}")
    print(f"{'='*60}\n")

    def make_env_phase():
        return CurriculumWrapper(
            LoveLetterRLEnv(num_players=4),
            agent_id="player_0",
            n_heuristics=n_heuristics,
        )

    dummy_env = make_env_phase()

    all_params = list(set(
        list(encoder.parameters()) + list(actor.parameters())
        + list(critic.parameters()) + list(belief_head.parameters())
    ))
    optim = torch.optim.AdamW(all_params, lr=lr)

    def dist_fn(logits):
        return Categorical(logits=logits)

    ppo_policy = AuxPPOPolicy(
        belief_head=belief_head,
        belief_coef=0.05,
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        discount_factor=0.99,
        action_space=dummy_env.action_space,
        action_scaling=False,
        action_bound_method="",
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        deterministic_eval=True,
    )

    train_envs = DummyVectorEnv([make_env_phase for _ in range(8)])
    test_envs = DummyVectorEnv([make_env_phase for _ in range(4)])
    train_collector = Collector(ppo_policy, train_envs, VectorReplayBuffer(20000, 8))
    test_collector = Collector(ppo_policy, test_envs)

    onpolicy_trainer(
        policy=ppo_policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=n_epochs,
        step_per_epoch=10000,
        repeat_per_collect=10,
        episode_per_test=100,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda r: r >= 1.5,  # objectif élevé
    )

    return ppo_policy


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("CURRICULUM LEARNING : 1H+2R → 2H+1R → 3H")
    print("=" * 60)

    # Charger le LEAGUE_CHAMPION
    encoder = RecurrentEncoder()
    actor = MaskedActor(encoder)
    critic = Critic(encoder)
    belief_head = BeliefHead(encoder)

    print(f"\nChargement de {STARTING_CHECKPOINT}...")
    ckpt = torch.load(STARTING_CHECKPOINT, map_location="cpu", weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    belief_head.load_state_dict(ckpt["belief_head"])

    # ===== ÉVAL INITIALE =====
    print("\n--- ÉVAL INITIALE (avant curriculum) ---")
    initial_results = evaluate_actor(actor, n_heuristics_list=[0, 1, 2, 3], n_games=300)
    print_eval("LEAGUE_CHAMPION (avant)", initial_results)

    # ===== PHASE 1 : 1H + 2R (consolidation) =====
    train_phase(actor, critic, encoder, belief_head,
                n_heuristics=1, n_epochs=30, label="Phase 1 (1H+2R)", lr=5e-5)
    torch.save({
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "belief_head": belief_head.state_dict(),
    }, checkpoint_path("curriculum_phase1.pth"))

    p1_results = evaluate_actor(actor, n_heuristics_list=[0, 1, 2, 3], n_games=300)
    print_eval("Après PHASE 1 (1H+2R)", p1_results)

    # ===== PHASE 2 : 2H + 1R =====
    train_phase(actor, critic, encoder, belief_head,
                n_heuristics=2, n_epochs=40, label="Phase 2 (2H+1R)", lr=3e-5)
    torch.save({
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "belief_head": belief_head.state_dict(),
    }, checkpoint_path("curriculum_phase2.pth"))

    p2_results = evaluate_actor(actor, n_heuristics_list=[0, 1, 2, 3], n_games=300)
    print_eval("Après PHASE 2 (2H+1R)", p2_results)

    # ===== PHASE 3 : 3H =====
    train_phase(actor, critic, encoder, belief_head,
                n_heuristics=3, n_epochs=50, label="Phase 3 (3H)", lr=2e-5)
    torch.save({
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "belief_head": belief_head.state_dict(),
    }, checkpoint_path("curriculum_phase3_final.pth"))

    p3_results = evaluate_actor(actor, n_heuristics_list=[0, 1, 2, 3], n_games=500)
    print_eval("Après PHASE 3 (3H) — FINAL", p3_results)

    # ===== RÉCAP =====
    print("\n" + "=" * 60)
    print("RÉCAPITULATIF DU CURRICULUM")
    print("=" * 60)
    print(f"\n{'Config':12} | {'Initial':>10} | {'Phase1':>10} | {'Phase2':>10} | {'Phase3':>10}")
    print("-" * 64)
    for cfg in ["vs_0H", "vs_1H", "vs_2H", "vs_3H"]:
        i = initial_results[cfg]["reward"]
        p1 = p1_results[cfg]["reward"]
        p2 = p2_results[cfg]["reward"]
        p3 = p3_results[cfg]["reward"]
        print(f"{cfg:12} | {i:10.3f} | {p1:10.3f} | {p2:10.3f} | {p3:10.3f}")
    print(f"\n Modèles sauvegardés dans : {checkpoint_path('curriculum_phase1.pth').parent}")


if __name__ == "__main__":
    main()
