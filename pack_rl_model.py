# ppo_pack_proto.py
# Minimal PPO prototype for Pack dynamic grouping
# Requirements: numpy, torch

import math
import random
import copy
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Repro
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Simple synthetic digital twin
# -------------------------
class SimpleDigitalTwin:
    """
    Evaluate assigned cells and output pack-level metrics.
    Replace/extend this with a calibrated ECM or data-driven twin for real deployment.
    """
    def __init__(self, current_a=1.0, dcr_safety_threshold=200.0):
        self.current = current_a
        self.dcr_threshold = dcr_safety_threshold

    def evaluate(self, assigned_cells):
        caps = np.array([c['cap'] for c in assigned_cells])
        dcrs = np.array([c['dcr'] for c in assigned_cells])
        sdrs = np.array([c['sdr'] for c in assigned_cells])
        vocs = np.array([c['voc'] for c in assigned_cells])
        pack_capacity = float(np.min(caps))  # series pack capacity limited by weakest cell
        pack_avg_voc = float(np.mean(vocs))
        mean_dcr = float(np.mean(dcrs))
        dynamic_voltage_drop = self.current * mean_dcr / 1000.0  # V (dcr in mOhm)
        lifetime_penalty = float(np.mean(sdrs) * 0.5 + mean_dcr * 0.001)
        safety_flag = bool(np.max(dcrs) > self.dcr_threshold)
        return {
            'pack_capacity_mAh': pack_capacity,
            'pack_avg_voc_V': pack_avg_voc,
            'dynamic_voltage_drop_V': dynamic_voltage_drop,
            'mean_dcr_mOhm': mean_dcr,
            'mean_sdr_pct_per_day': float(np.mean(sdrs)),
            'lifetime_penalty': lifetime_penalty,
            'safety_flag': safety_flag
        }

# -------------------------
# Environment
# -------------------------
class PackEnv:
    """
    Pack filling environment:
      - candidates: list of cell dicts (cap, dcr, sdr, voc)
      - action: choose one candidate index (global index within candidate pool)
      - we use action masking: only unassigned candidate indices are allowed
    State returns a dict: candidate_features (M,4), mask (M,), pack_meta
    """
    def __init__(self, candidates, pack_size=4, target_capacity=3000.0, twin=None):
        self.orig_candidates = copy.deepcopy(candidates)
        self.pack_size = pack_size
        self.target_capacity = target_capacity
        self.twin = twin if twin is not None else SimpleDigitalTwin()
        self.reset()

    def reset(self):
        self.candidates = copy.deepcopy(self.orig_candidates)
        random.shuffle(self.candidates)
        self.unassigned = list(range(len(self.candidates)))
        self.assigned = []
        self.done = False
        return self._get_state()

    def _get_state(self):
        M = len(self.candidates)
        caps = np.array([c['cap'] for c in self.candidates], dtype=np.float32)
        dcrs = np.array([c['dcr'] for c in self.candidates], dtype=np.float32)
        sdrs = np.array([c['sdr'] for c in self.candidates], dtype=np.float32)
        vocs = np.array([c['voc'] for c in self.candidates], dtype=np.float32)
        # simple normalization (mean/std)
        def norm(x):
            xm = x.mean() if x.size>0 else 0.0
            xs = x.std() if x.size>0 else 1.0
            xs = xs if xs>1e-6 else 1.0
            return (x - xm) / xs
        nf = np.stack([norm(caps), norm(dcrs), norm(sdrs), norm(vocs)], axis=1)
        mask = np.zeros((M,), dtype=np.float32)
        mask[self.unassigned] = 1.0
        if len(self.assigned) > 0:
            acaps = np.array([self.candidates[i]['cap'] for i in self.assigned], dtype=np.float32)
            adcrs = np.array([self.candidates[i]['dcr'] for i in self.assigned], dtype=np.float32)
            asdrs = np.array([self.candidates[i]['sdr'] for i in self.assigned], dtype=np.float32)
            pack_mean = np.array([acaps.mean(), adcrs.mean(), asdrs.mean()], dtype=np.float32)
            pack_var = np.array([acaps.var(), adcrs.var(), asdrs.var()], dtype=np.float32)
        else:
            pack_mean = np.zeros((3,), dtype=np.float32)
            pack_var = np.zeros((3,), dtype=np.float32)
        pack_count = np.array([len(self.assigned)], dtype=np.float32)
        state = {
            'candidate_features': nf,
            'mask': mask,
            'pack_meta': np.concatenate([pack_count, pack_mean, pack_var, np.array([self.target_capacity], dtype=np.float32)])
        }
        return state

    def step(self, action_index):
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")
        if action_index < 0 or action_index >= len(self.unassigned):
            self.done = True
            return self._get_state(), -10.0, True, {'invalid_action': True}
        chosen_idx = self.unassigned.pop(action_index)
        self.assigned.append(chosen_idx)
        reward = self._intermediate_reward()
        if len(self.assigned) >= self.pack_size:
            assigned_cells = [self.candidates[i] for i in self.assigned]
            metrics = self.twin.evaluate(assigned_cells)
            term_reward = self._terminal_reward(metrics)
            self.done = True
            return self._get_state(), reward + term_reward, True, {'metrics': metrics}
        else:
            return self._get_state(), reward, False, {}

    def _intermediate_reward(self):
        if len(self.assigned) <= 1:
            return 0.0
        acaps = np.array([self.candidates[i]['cap'] for i in self.assigned], dtype=np.float32)
        adcrs = np.array([self.candidates[i]['dcr'] for i in self.assigned], dtype=np.float32)
        asdrs = np.array([self.candidates[i]['sdr'] for i in self.assigned], dtype=np.float32)
        def norm_var(x):
            m = max(1e-3, x.mean())
            return x.var() / (m*m)
        score = - (0.5 * norm_var(acaps) + 0.3 * norm_var(adcrs) + 0.2 * norm_var(asdrs))
        return float(score)

    def _terminal_reward(self, metrics):
        cap = metrics['pack_capacity_mAh']
        vdrop = metrics['dynamic_voltage_drop_V']
        safety = metrics['safety_flag']
        lifetime_pen = metrics['lifetime_penalty']
        cap_score = - abs(cap - self.target_capacity) / max(1.0, self.target_capacity)
        vdrop_score = - vdrop
        life_score = -0.2 * lifetime_pen
        safety_score = -5.0 if safety else 0.0
        total = 2.0 * cap_score + 1.0 * vdrop_score + life_score + safety_score
        return float(total)

# -------------------------
# Actor-Critic and PPO utilities
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

Transition = namedtuple('Transition', ['state', 'action', 'logp', 'reward', 'done', 'value'])

def flatten_state(state, max_m):
    nf = state['candidate_features']
    mask = state['mask']
    pack_meta = state['pack_meta']
    M = nf.shape[0]
    pad = max_m - M
    if pad < 0:
        raise ValueError("max_m too small")
    nf_flat = np.concatenate([nf.flatten(), np.zeros(pad * nf.shape[1], dtype=np.float32)])
    mask_flat = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
    flat = np.concatenate([nf_flat, mask_flat, pack_meta])
    return flat.astype(np.float32)

def masked_softmax_logits(logits, mask, eps=-1e9):
    masked = logits.clone()
    inv_mask = (mask <= 0.5)
    masked[inv_mask] = eps
    maxl = masked.max().detach()
    exps = torch.exp(masked - maxl)
    exps_masked = exps * mask
    sum_exps = exps_masked.sum() + 1e-8
    probs = exps_masked / sum_exps
    logp = torch.log(probs + 1e-8)
    return probs, logp

class PPOTrainer:
    def __init__(self, obs_dim, action_dim, max_candidates, hidden_dim=128, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = ActorCritic(obs_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_candidates = max_candidates

    def select_action(self, state_flat, mask):
        state_t = torch.tensor(state_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(state_t)
        logits = logits[0]
        value = value.item()
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
        probs, logp = masked_softmax_logits(logits, mask_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        logp_val = logp[action].item()
        return action, logp_val, value

    def compute_gae(self, trans, last_value=0.0):
        rewards = [t.reward for t in trans]
        values = [t.value for t in trans] + [last_value]
        dones = [t.done for t in trans]
        gae = 0.0
        returns = []
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1.0 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1.0 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return returns, advantages

    def update(self, transitions, epochs=4, batch_size=64):
        obs = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        old_logps = np.array([t.logp for t in transitions], dtype=np.float32)
        returns = np.array([t.reward for t in transitions], dtype=np.float32)
        advantages = np.array([t.value for t in transitions], dtype=np.float32)
        N = len(transitions)
        inds = np.arange(N)
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, N, batch_size):
                mb_inds = inds[start:start+batch_size]
                mb_obs = torch.tensor(obs[mb_inds], dtype=torch.float32, device=self.device)
                mb_actions = torch.tensor(actions[mb_inds], dtype=torch.long, device=self.device)
                mb_old_logps = torch.tensor(old_logps[mb_inds], dtype=torch.float32, device=self.device)
                mb_returns = torch.tensor(returns[mb_inds], dtype=torch.float32, device=self.device)
                mb_advs = torch.tensor(advantages[mb_inds], dtype=torch.float32, device=self.device)
                logits, values = self.net(mb_obs)
                max_m = self.max_candidates
                feat_per = 4
                nf_len = max_m * feat_per
                mask_tensor = mb_obs[:, nf_len:nf_len+max_m]
                probs = []
                logps = []
                for i in range(mb_obs.shape[0]):
                    pmask = mask_tensor[i]
                    logits_i = logits[i]
                    p, logp_all = masked_softmax_logits(logits_i, pmask)
                    probs.append(p.unsqueeze(0))
                    logps.append(logp_all.unsqueeze(0))
                probs = torch.cat(probs, dim=0)
                logps_all = torch.cat(logps, dim=0)
                mb_new_logps = logps_all.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(mb_new_logps - mb_old_logps)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * mb_advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean((mb_returns - values)**2)
                entropy = -torch.mean((probs * torch.log(probs + 1e-8)).sum(dim=1))
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

# -------------------------
# Synthetic data and training demo
# -------------------------
def generate_candidates(M, base_cap=3000, cap_var=150, base_dcr=60, dcr_var=10, base_sdr=0.05, sdr_var=0.02, base_voc=3.7):
    candidates = []
    for i in range(M):
        cap = np.random.normal(base_cap, cap_var)
        dcr = np.random.normal(base_dcr, dcr_var)
        sdr = max(0.0, np.random.normal(base_sdr, sdr_var))
        voc = np.random.normal(base_voc, 0.02)
        candidates.append({'cap': float(max(500, cap)),
                           'dcr': float(max(1.0, dcr)),
                           'sdr': float(sdr),
                           'voc': float(voc),
                           'id': i})
    return candidates

def train_demo(num_episodes=400, max_candidates=12, pack_size=4):
    sample_env = PackEnv(generate_candidates(max_candidates), pack_size=pack_size)
    sample_state = sample_env.reset()
    obs_dim = flatten_state(sample_state, max_candidates).shape[0]
    action_dim = max_candidates
    trainer = PPOTrainer(obs_dim, action_dim, max_candidates, hidden_dim=128, lr=3e-4)
    all_episode_rewards = []
    for ep in range(num_episodes):
        cand_pool = generate_candidates(max_candidates)
        env = PackEnv(cand_pool, pack_size=pack_size)
        state = env.reset()
        done = False
        transitions = []
        ep_reward = 0.0
        while not done:
            state_flat = flatten_state(state, max_candidates)
            # compute logits and sample among valid indices
            with torch.no_grad():
                state_t = torch.tensor(state_flat, dtype=torch.float32, device=trainer.device).unsqueeze(0)
                logits, val = trainer.net(state_t)
                logits = logits[0]
                mask_t = torch.tensor(state['mask'], dtype=torch.float32, device=trainer.device)
                probs, logp_all = masked_softmax_logits(logits, mask_t)
                dist = torch.distributions.Categorical(probs)
                full_action = dist.sample().item()
                logp_val = logp_all[full_action].item()
                value = val.item()
            # map chosen global action to index in unassigned list
            if full_action not in env.unassigned:
                chosen_global = random.choice(env.unassigned)
                action_index = env.unassigned.index(chosen_global)
            else:
                action_index = env.unassigned.index(full_action)
            next_state, reward, done, info = env.step(action_index)
            ep_reward += reward
            transitions.append(Transition(state=flatten_state(state, max_candidates), action=full_action, logp=logp_val, reward=reward, done=done, value=value))
            state = next_state
        # GAE & update
        last_value = 0.0
        returns, advantages = trainer.compute_gae(transitions, last_value=last_value)
        upd_trans = []
        for t, R, A in zip(transitions, returns, advantages):
            upd_trans.append(Transition(state=t.state, action=t.action, logp=t.logp, reward=R, done=t.done, value=A))
        trainer.update(upd_trans, epochs=6, batch_size=64)
        all_episode_rewards.append(ep_reward)
        if (ep+1) % 50 == 0:
            avg_r = np.mean(all_episode_rewards[-50:])
            print(f"Episode {ep+1}/{num_episodes}, avg reward (last50)={avg_r:.3f}")
    # save model
    torch.save(trainer.net.state_dict(), "ppo_pack_model.pth")
    print("Model saved to ppo_pack_model.pth")
    return trainer, all_episode_rewards

if __name__ == "__main__":
    trainer, rewards = train_demo(num_episodes=300, max_candidates=12, pack_size=4)
    # quick evaluation
    def evaluate(trainer, n=30):
        vals = []
        for _ in range(n):
            env = PackEnv(generate_candidates(12), pack_size=4)
            s = env.reset()
            done = False
            rsum = 0.0
            while not done:
                sf = flatten_state(s, 12)
                with torch.no_grad():
                    st = torch.tensor(sf, dtype=torch.float32, device=trainer.device).unsqueeze(0)
                    logits, val = trainer.net(st)
                    logits = logits[0]
                    mask_t = torch.tensor(s['mask'], dtype=torch.float32, device=trainer.device)
                    probs, logp_all = masked_softmax_logits(logits, mask_t)
                    a = torch.argmax(probs).item()
                if a not in env.unassigned:
                    chosen_global = random.choice(env.unassigned)
                    action_index = env.unassigned.index(chosen_global)
                else:
                    action_index = env.unassigned.index(a)
                s, r, done, info = env.step(action_index)
                rsum += r
            vals.append(rsum)
        return vals
    ev = evaluate(trainer, n=50)
    print("Eval mean:", np.mean(ev), "std:", np.std(ev))
