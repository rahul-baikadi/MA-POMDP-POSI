"""
MA-PDOL-NoRec: Multi-Agent PDOL with Shared Counts but NO Recommendations
==========================================================================

This is the MA-PDOL-Sharing algorithm with ONE modification:
  - Cross-agent coordinate recommendations (Eq. 22-23) are REMOVED.

What IS shared (same as MA-PDOL-Sharing):
  1. All agents write/read from a SINGLE shared UCB-VI transition model.
  2. Global weights are shared and updated with all agents' data (Eq. 21).

What is removed:
  - Agents do NOT exchange "best coordinate" recommendations at epoch end.
  - The augmented set is just base_set ∪ {leader} (no recommended set).

This isolates the effect of the recommendation mechanism while keeping
the benefits of shared exploration counts and shared weight updates.

Usage:
    python MA-PDOL-NoRec.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import math
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


def safe_exp(x: float, cap: float = 500.0) -> float:
    """Exponentiate with overflow protection."""
    return math.exp(min(x, cap))


###########################################################################
# Subclass 2 POSI-POMDP Environment (identical to MA-PDOL.py)
###########################################################################

@dataclass
class POSIPOMDPSubclass2:
    """
    Episodic POSI-POMDP under Subclass 2 dynamics.
    See MA-PDOL.py for full documentation.
    """
    d: int
    S_tilde_size: int
    A: int
    H: int
    P: np.ndarray = field(default=None, repr=False)
    r: np.ndarray = field(default=None, repr=False)
    delta1: np.ndarray = field(default=None, repr=False)
    _P_cdf: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        assert self.d >= 3 and self.S_tilde_size >= 2
        assert self.A >= 2 and self.H >= 1
        if self.P is None:
            self._random_init()

    def _random_init(self):
        St = self.S_tilde_size
        self.P = np.random.dirichlet(np.ones(St),
                                     size=(self.H, self.d, St, self.A))
        self.r = np.random.uniform(0, 1, size=(self.H, St, self.A))
        self.delta1 = np.random.dirichlet(np.ones(St), size=(self.d,))
        self._P_cdf = np.cumsum(self.P, axis=-1)

    def clone(self):
        e = POSIPOMDPSubclass2.__new__(POSIPOMDPSubclass2)
        for attr in ('d', 'S_tilde_size', 'A', 'H', 'P', 'r', 'delta1', '_P_cdf'):
            setattr(e, attr, getattr(self, attr))
        return e

    def reset(self) -> np.ndarray:
        u = np.random.random(self.d)
        cdf = np.cumsum(self.delta1, axis=-1)
        return np.array([np.searchsorted(cdf[i], u[i])
                         for i in range(self.d)], dtype=np.int32)

    def step(self, state: np.ndarray, action: int, h: int) -> np.ndarray:
        cdf = self._P_cdf[h, np.arange(self.d), state, action]
        u = np.random.random(self.d)
        ns = np.sum(u[:, None] >= cdf, axis=1).astype(np.int32)
        return np.clip(ns, 0, self.S_tilde_size - 1)

    def compute_optimal_value(self) -> Tuple[float, int, np.ndarray]:
        St = self.S_tilde_size
        V_all = np.zeros(self.d)
        best_V, best_i = -np.inf, 0
        for i in range(self.d):
            V = np.zeros((self.H + 1, St))
            for h in range(self.H - 1, -1, -1):
                ev = np.einsum('xas,s->xa', self.P[h, i], V[h + 1])
                V[h] = np.max(self.r[h] + ev, axis=1)
            Vi = self.delta1[i] @ V[0]
            V_all[i] = Vi
            if Vi > best_V:
                best_V = Vi; best_i = i
        return best_V, best_i, V_all


###########################################################################
# UCB-VI (identical to MA-PDOL.py)
###########################################################################

class CoordinateUCBVI:
    """Coordinate-wise UCB-VI with vectorised backward induction."""

    def __init__(self, d: int, St: int, A: int, H: int, delta: float = 0.05):
        self.d = d; self.St = St; self.A = A; self.H = H; self.delta = delta
        self.counts = np.zeros((H, d, St, A), dtype=np.float64)
        self.counts_next = np.zeros((H, d, St, A, St), dtype=np.float64)

    def update(self, h: int, i: int, x: int, a: int, x_next: int):
        self.counts[h, i, x, a] += 1
        self.counts_next[h, i, x, a, x_next] += 1

    def compute_Q_batch(self, coords: list, r_func: np.ndarray,
                        K: int) -> Dict[int, np.ndarray]:
        St = self.St
        log_t = math.log(max(2.0, self.H * St * self.A * K / self.delta))
        result = {}
        for ci in coords:
            Q = np.zeros((self.H, St, self.A))
            V = np.zeros((self.H + 1, St))
            for h in range(self.H - 1, -1, -1):
                rem = float(self.H - h)
                n_arr = self.counts[h, ci]
                n_safe = np.maximum(n_arr, 1.0)
                p_hat = self.counts_next[h, ci] / n_safe[:, :, None]
                p_hat[n_arr == 0] = 1.0 / St
                bonus = np.minimum(rem,
                    np.sqrt(rem * log_t / n_safe) + rem * log_t / n_safe)
                ev = np.einsum('xas,s->xa', p_hat, V[h + 1])
                Q[h] = np.minimum(rem, r_func[h] + ev + bonus)
                V[h] = np.minimum(rem, np.max(Q[h], axis=1))
            result[ci] = Q
        return result


###########################################################################
# Agent State (simplified -- no recommendations)
###########################################################################

@dataclass
class AgentState:
    agent_id: int
    base_set: np.ndarray
    leader: int = -1
    augmented_set: np.ndarray = None
    non_leader: np.ndarray = None
    fresh_pool: list = field(default_factory=list)
    local_weights: dict = field(default_factory=dict)


###########################################################################
# MA-PDOL-NoRec Algorithm
###########################################################################

class MAPDOLNoRec:
    """
    MA-PDOL with shared UCB counts but NO inter-agent recommendations.

    Differences from MA-PDOL-Sharing:
      - Recommendations (Eq. 22-23) are removed; agents never exchange
        coordinate tips, so augmented_set = base_set ∪ {leader} only.
    Everything else (shared UCB, shared global weights, Eq. 21) is identical.
    """

    def __init__(self, env: POSIPOMDPSubclass2, N: int, d_tilde: int,
                 K: int, delta: float = 0.05):
        assert N >= 1 and K >= 1
        max_dt = env.d // N - 1
        assert max_dt >= 2, (
            f"Cannot satisfy partial observability: d/N - 1 = "
            f"{max_dt} < 2. Increase d or decrease N "
            f"(need d >= 3*N, got d={env.d}, N={N}).")
        if d_tilde > max_dt:
            print(f"  [Auto-adjust] d_tilde={d_tilde} exceeds "
                  f"d/N-1={max_dt}. Setting d_tilde={max_dt} "
                  f"to enforce d_tilde*N < d.")
            d_tilde = max_dt

        self.env = env
        self.N = N
        self.d = env.d
        self.dt = d_tilde
        self.H = env.H
        self.A = env.A
        self.K = K
        self.St = env.S_tilde_size
        self.delta = delta

        # Epoch length -- Eq. (1)
        self.epoch_len = max(1, int(3*np.ceil(
            self.d / (N * d_tilde) + (N - 1) / d_tilde)))

        # Learning rates -- Eq. (6), (7)
        self.eta1 = min(1.0, 1.0 / math.sqrt(K))
        C = 2
        eff_size = math.ceil(self.d / N) + (N - 1)
        self.eta2 = min(C * eff_size / max(1, d_tilde - 1) * self.eta1,
                d_tilde / max(1, self.H), 1.0)

        # Gamma -- Eq. (20)
        self.Gamma = math.ceil(self.d / N) + N - 2

        # SHARED global weights (all agents read/write same array)
        self.global_weights = np.ones(self.d, dtype=np.float64)

        # SHARED UCB-VI (all agents write/read from same counts)
        shared_ucb_module = CoordinateUCBVI(
            self.d, self.St, self.A, self.H, delta)
        self.ucb = [shared_ucb_module] * N   # all N entries → same object

        # Initialise agents -- Eq. (2), (3)  (no recommendations)
        base_size = max(1, math.ceil(self.d / self.N))
        self.agents = []
        for n in range(N):
            start = n * base_size
            end = min(start + base_size, self.d)
            S_n = np.array(list(range(start, end)) if start < self.d
                           else [], dtype=int)
            self.agents.append(AgentState(n, S_n))

    # ----- Helper methods (recommendations removed) -----

    def _global_distribution(self) -> np.ndarray:
        """Eq. (5) — shared weights."""
        w = np.clip(self.global_weights, 1e-300, 1e300)
        s = w.sum()
        if s == 0 or not np.isfinite(s):
            return np.ones(self.d) / self.d
        p = (1 - self.eta1) * (w / s) + self.eta1 / self.d
        p = np.clip(p, 1e-15, None)
        return p / p.sum()

    def _form_augmented_set(self, ag: AgentState, leader: int):
        """Eq. (9)-(11) — no recommendations merged in."""
        ag.leader = leader
        # Only base_set + leader; no recommended set
        aug = set(ag.base_set.tolist()) | {leader}
        ag.augmented_set = np.array(sorted(aug), dtype=int)
        ag.non_leader = np.array(sorted(aug - {leader}), dtype=int)
        ag.fresh_pool = list(ag.non_leader)

    def _form_query_set(self, ag: AgentState) -> np.ndarray:
        """Eq. (12) with follower rotation."""
        need = self.dt - 1
        B = list(ag.non_leader)
        U = ag.fresh_pool

        if not B:
            F = [j for j in range(self.d) if j != ag.leader][:need]
        elif len(U) >= need:
            ix = np.random.choice(len(U), need, replace=False)
            F = [U[i] for i in ix]
            ag.fresh_pool = [U[j] for j in range(len(U))
                             if j not in set(ix)]
        else:
            F = list(U)
            still = need - len(F)
            comp = [b for b in B if b not in set(U)]
            if still > 0 and comp:
                fi = np.random.choice(len(comp), min(still, len(comp)),
                                      replace=False)
                F.extend([comp[j] for j in fi])
            still = need - len(F)
            if still > 0:
                F.extend([b for b in B if b not in set(F)][:still])
            ag.fresh_pool = []

        I_k = np.unique(np.concatenate([[ag.leader], F]))
        if len(I_k) < self.dt:
            rem = [j for j in range(self.d) if j not in set(I_k)]
            nd = self.dt - len(I_k)
            if rem:
                I_k = np.unique(np.concatenate(
                    [I_k, np.random.choice(rem, min(nd, len(rem)),
                                           replace=False)]))
        return I_k[:self.dt].astype(int)

    def _local_distribution(self, ag: AgentState,
                            I_k: list) -> np.ndarray:
        """Eq. (14)"""
        vals = np.array([ag.local_weights.get(i, 1e-300) for i in I_k])
        s = vals.sum()
        if s > 0 and np.isfinite(s):
            pl = (1 - self.eta2) * (vals / s) + self.eta2 / self.dt
            pl /= pl.sum()
        else:
            pl = np.ones(len(I_k)) / len(I_k)
        return pl

    # ----- Main loop -----

    def run(self, verbose: bool = False) -> dict:
        """
        Run MA-PDOL with shared counts but no recommendations.
        Agents share UCB and global weights, but never exchange
        coordinate recommendations.
        """
        V_star, best_coord, V_all = self.env.compute_optimal_value()
        episode_count = 0
        all_rewards = {n: [] for n in range(self.N)}
        cum_regret = {n: [] for n in range(self.N)}

        while episode_count < self.K:
            # --- Epoch start: each agent picks leader from shared weights ---
            p_global = self._global_distribution()
            for n in range(self.N):
                leader = int(np.random.choice(self.d, p=p_global))
                self._form_augmented_set(self.agents[n], leader)

            epoch_data = {n: [] for n in range(self.N)}

            for ep_in_epoch in range(self.epoch_len):
                if episode_count >= self.K:
                    break
                episode_count += 1

                for n in range(self.N):
                    ag = self.agents[n]
                    I_k = self._form_query_set(ag)
                    I_list = [int(c) for c in I_k]

                    if ep_in_epoch == 0:
                        ag.local_weights = {
                            int(i): max(1e-300, self.global_weights[i])
                            for i in ag.augmented_set}

                    pl = self._local_distribution(ag, I_list)

                    # Q computed from PRIVATE counts
                    Q_cache = self.ucb[n].compute_Q_batch(
                        I_list, self.env.r, self.K)

                    state = self.env.reset()
                    ep_reward = 0.0
                    coord_rewards = np.zeros(len(I_list))
                    baseline_gains = np.zeros(len(I_list))

                    for h in range(self.H):
                        j_ctrl = np.random.choice(len(I_list), p=pl)
                        i_ctrl = I_list[j_ctrl]

                        q = Q_cache[i_ctrl][h, state[i_ctrl]]
                        mx = np.max(q)
                        best_acts = np.where(np.abs(q - mx) < 1e-10)[0]
                        action = int(np.random.choice(best_acts))

                        next_state = self.env.step(state, action, h)

                        leader_r = float(
                            self.env.r[h, state[ag.leader], action])
                        for j, i in enumerate(I_list):
                            ri = float(self.env.r[h, state[i], action])
                            coord_rewards[j] += ri
                            baseline_gains[j] += (ri - leader_r)
                            # Writes to SHARED counts
                            self.ucb[n].update(
                                h, i, state[i], action, next_state[i])

                        ep_reward += float(
                            self.env.r[h, state[i_ctrl], action])
                        state = next_state

                    for j, i in enumerate(I_list):
                        ag.local_weights[i] = min(
                            ag.local_weights.get(i, 1.0)
                            * safe_exp(self.eta2 / self.dt
                                       * coord_rewards[j]),
                            1e300)

                    epoch_data[n].append({
                        I_list[j]: baseline_gains[j]
                        for j in range(len(I_list))
                        if baseline_gains[j] != 0})

                    all_rewards[n].append(ep_reward)
                    prev = cum_regret[n][-1] if cum_regret[n] else 0.0
                    cum_regret[n].append(prev + (V_star - ep_reward))

                if verbose and episode_count % max(1, self.K // 10) == 0:
                    avg_r = np.mean([all_rewards[n][-1]
                                     for n in range(self.N)])
                    avg_reg = np.mean([cum_regret[n][-1]
                                       for n in range(self.N)])
                    print(f"  Ep {episode_count}/{self.K}  "
                          f"rew={avg_r:.3f}  cum_reg={avg_reg:.0f}")

            # --- Epoch-end: SHARED global weight update (Eq. 21) ---
            # All agents' data aggregated into single shared weights
            denom = self.d * max(1, self.dt - 1)
            for i in range(self.d):
                total_gain = sum(
                    ed.get(i, 0.0)
                    for n in range(self.N) for ed in epoch_data[n])
                if total_gain != 0:
                    exponent = np.clip(
                        (self.Gamma * self.eta1) / denom * total_gain,
                        -500, 500)
                    self.global_weights[i] = np.clip(
                        self.global_weights[i] * safe_exp(exponent),
                        1e-300, 1e300)

            # --- NO recommendations exchanged (Eq. 22-23 removed) ---

        return {
            'rewards': all_rewards,
            'cumulative_regret': cum_regret,
            'V_star': V_star,
            'best_coordinate': best_coord,
            'V_star_all': V_all,
            'total_episodes': episode_count,
            'eta1': self.eta1,
            'eta2': self.eta2,
        }


###########################################################################
# Plotting
###########################################################################

def plot_per_agent_regret(results_dict: dict, save_path: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
              '#911eb4', "#249cb7a6", '#f032e6', '#bfef45']
    K_max = 0
    for idx, (label, res) in enumerate(results_dict.items()):
        Na = len(res['rewards'])
        K = res['total_episodes']; K_max = max(K_max, K)
        ep = np.arange(1, K + 1)
        avg = np.mean([res['cumulative_regret'][n]
                        for n in range(Na)], axis=0)
        ax.plot(ep, avg, color=colors[idx % len(colors)],
                lw=2.2, label=label)

    ep = np.arange(1, K_max + 1)
    biggest = max(
        np.mean([r['cumulative_regret'][n][-1]
                 for n in range(len(r['rewards']))])
        for r in results_dict.values())
    ax.plot(ep, biggest / math.sqrt(K_max) * np.sqrt(ep),
            'k--', alpha=0.3, lw=1.5, label=r'$O(\sqrt{K})$ ref.')

    ax.set_xlabel('Episode (K)', fontsize=13)
    ax.set_ylabel('Per-Agent Cumulative Regret', fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


###########################################################################
# Main
###########################################################################

def main():
    # ---- Configurable parameters ----
    d = 40; S_tilde = 6; A = 4; H = 7; d_tilde = 4; K = 20000
    N_values = [5, 8]

    print("=" * 65)
    print("MA-PDOL-NoRec — Per-Agent Regret (Shared Counts, No Recommendations)")
    print("=" * 65)
    print(f"Environment: d={d}, |S~|={S_tilde}, A={A}, H={H}")
    print(f"Algorithm:   d~={d_tilde}, K={K}")
    print(f"Agents:      N = {N_values}")

    np.random.seed(42)
    base_env = POSIPOMDPSubclass2(d=d, S_tilde_size=S_tilde, A=A, H=H)
    V_star, best_i, _ = base_env.compute_optimal_value()
    print(f"V* = {V_star:.4f} (best coord i*={best_i})")
    print("-" * 65)

    results = {}
    for N_ in N_values:
        np.random.seed(0)
        env = base_env.clone()
        algo = MAPDOLNoRec(env=env, N=N_, d_tilde=d_tilde, K=K)
        label = f"MA-PDOL-NoRec N={N_}"
        print(f"\nRunning {label} (eta2={algo.eta2:.3f}, "
              f"epoch_len={algo.epoch_len}) ...")
        t0 = time.time()
        res = algo.run(verbose=True)
        elapsed = time.time() - t0
        avg_reg = np.mean([res['cumulative_regret'][n][-1]
                           for n in range(N_)])
        print(f"  Done in {elapsed:.1f}s  |  regret={avg_reg:.0f}  "
              f"reg/K={avg_reg/K:.4f}")
        results[label] = res

    # ---- Summary table ----
    print("\n" + "=" * 65)
    print(f"{'Config':<25} {'Regret':>8} {'Reg/K':>8} "
          f"{'Reg/sqK':>10} {'Last200':>9}")
    print("-" * 65)
    for label, res in results.items():
        Na = len(res['rewards'])
        ar = np.mean([res['cumulative_regret'][n][-1] for n in range(Na)])
        lr = np.mean([np.mean(res['rewards'][n][-200:]) for n in range(Na)])
        print(f"{label:<25} {ar:>8.0f} {ar/K:>8.4f} "
              f"{ar/math.sqrt(K):>10.2f} {lr:>9.4f}")

    # ---- Plot ----
    plot_per_agent_regret(
        results,
        f'MA-PDOL_norec_regret_{datetime.now().strftime("%m%d_%H%M")}.png',
        f'MA-PDOL-NoRec — Per-Agent Regret vs Episode\n'
        f'(d={d}, d̃={d_tilde}, |S̃|={S_tilde}, A={A}, H={H})')


if __name__ == "__main__":
    main()