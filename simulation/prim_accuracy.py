"""
PRI Method Simulation Study (Accuracy Test)
"""

# =============================================================================
# Setup
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns
from tqdm import tqdm
import warnings
import time
import collections
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("PRI Method SIMULATION STUDY for ΔAccuracy")
print("="*70)

# =============================================================================
# Utility Functions
# =============================================================================

def generate_contingency_table(n, p10, p01, p11=None):
    """Generate 2x2 contingency table from multinomial."""
    assert 0 <= p10 <= 1 and 0 <= p01 <= 1
    assert p10 + p01 <= 1

    if p11 is None:
        remaining = 1 - p10 - p01
        p11 = remaining * 0.6
        p00 = remaining * 0.4
    else:
        p00 = 1 - p11 - p10 - p01

    counts = np.random.multinomial(n, [p11, p10, p01, p00])
    return counts[0], counts[1], counts[2], counts[3]


def generate_stratified_data(N_pos, N_neg, p10_pos, p01_pos, p10_neg, p01_neg,
                             p11_pos=None, p11_neg=None):
    """Generate stratified paired binary data."""
    n11_pos, n10_pos, n01_pos, n00_pos = generate_contingency_table(
        N_pos, p10_pos, p01_pos, p11=p11_pos)
    n11_neg, n10_neg, n01_neg, n00_neg = generate_contingency_table(
        N_neg, p10_neg, p01_neg, p11=p11_neg)
    return {
        'N_pos': N_pos, 'N_neg': N_neg, 'N': N_pos + N_neg,
        'n10_pos': n10_pos, 'n01_pos': n01_pos,
        'n10_neg': n10_neg, 'n01_neg': n01_neg,
        'n10': n10_pos + n10_neg, 'n01': n01_pos + n01_neg
    }


def params_from_effect(disc_rate, effect):
    """Convert (disc_rate, effect) to (p10, p01)."""
    p10 = (disc_rate + effect) / 2
    p01 = (disc_rate - effect) / 2
    if p10 < 0 or p01 < 0 or p10 > 1 or p01 > 1:
        raise ValueError(
            f"Invalid: disc={disc_rate}, effect={effect} -> p10={p10:.4f}, p01={p01:.4f}"
        )
    return p10, p01


def ab_sq_to_ab(ab_sq):
    """
    Convert (a-b)² to (a, b) where b = -a (balanced null).
    (a-b)² = (2a)² = 4a²  =>  a = sqrt(ab_sq)/2
    """
    a = np.sqrt(ab_sq) / 2
    b = -a
    return round(a, 4), round(b, 4)


def mcnemar_test(data):
    """Original McNemar test using pooled variance."""
    n10, n01, N = data['n10'], data['n01'], data['N']
    delta = (n10 - n01) / N
    if n10 + n01 == 0:
        return {'delta': delta, 'var': np.nan, 'z': 0, 'p_value': 1.0}
    var = (n10 + n01) / (N ** 2)
    z = delta / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {'delta': delta, 'var': var, 'z': z, 'p_value': p_value}


def stratified_test(data):
    """Proposed stratified test using within-stratum variance."""
    N_pos, N_neg, N = data['N_pos'], data['N_neg'], data['N']
    n10_pos, n01_pos = data['n10_pos'], data['n01_pos']
    n10_neg, n01_neg = data['n10_neg'], data['n01_neg']

    w_pos, w_neg = N_pos / N, N_neg / N
    delta_R = (n10_pos - n01_pos) / N_pos if N_pos > 0 else 0
    delta_S = (n10_neg - n01_neg) / N_neg if N_neg > 0 else 0
    delta   = w_pos * delta_R + w_neg * delta_S

    if N_pos > 0 and (n10_pos + n01_pos) > 0:
        var_R = (n10_pos + n01_pos) / (N_pos**2) - ((n10_pos - n01_pos)**2) / (N_pos**3)
        var_R = max(var_R, 0)
    else:
        var_R = 0

    if N_neg > 0 and (n10_neg + n01_neg) > 0:
        var_S = (n10_neg + n01_neg) / (N_neg**2) - ((n10_neg - n01_neg)**2) / (N_neg**3)
        var_S = max(var_S, 0)
    else:
        var_S = 0

    var = (w_pos**2) * var_R + (w_neg**2) * var_S
    if var <= 0:
        return {'delta': delta, 'var': np.nan, 'z': 0, 'p_value': 1.0}

    z       = delta / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {'delta': delta, 'var': var, 'z': z, 'p_value': p_value}


def run_simulation_scenario(N, ratio, disc_pos, disc_neg, a, b,
                            n_sim=10000, alpha=0.05, seed=None,
                            p11_pos=None, p11_neg=None):
    """Run simulation for a single scenario."""
    if seed is not None:
        np.random.seed(seed)

    N_pos = max(int(round(N * ratio / (1 + ratio))), 1)
    N_neg = max(N - N_pos, 1)

    try:
        p10_pos, p01_pos = params_from_effect(disc_pos, a)
        p10_neg, p01_neg = params_from_effect(disc_neg, b)
    except ValueError as e:
        print(f"  Skipping: {e}")
        return None

    mcn_rejects   = 0
    strat_rejects = 0
    mcn_vars      = []
    strat_vars    = []
    var_ratios    = []

    for _ in range(n_sim):
        data  = generate_stratified_data(N_pos, N_neg,
                                         p10_pos, p01_pos,
                                         p10_neg, p01_neg,
                                         p11_pos=p11_pos,
                                         p11_neg=p11_neg)
        mcn   = mcnemar_test(data)
        strat = stratified_test(data)

        if mcn['p_value']   < alpha: mcn_rejects   += 1
        if strat['p_value'] < alpha: strat_rejects += 1

        if not np.isnan(mcn['var']):
            mcn_vars.append(mcn['var'])
        if not np.isnan(strat['var']) and strat['var'] > 0:
            strat_vars.append(strat['var'])
            if mcn['var'] > 0:
                var_ratios.append(mcn['var'] / strat['var'])

    mcn_rate   = mcn_rejects   / n_sim
    strat_rate = strat_rejects / n_sim

    return {
        'N': N, 'N_pos': N_pos, 'N_neg': N_neg, 'ratio': ratio,
        'disc_pos': disc_pos, 'disc_neg': disc_neg, 'a': a, 'b': b,
        'mcnemar_reject_rate':   mcn_rate,
        'stratified_reject_rate': strat_rate,
        'mcnemar_se':   np.sqrt(mcn_rate   * (1 - mcn_rate)   / n_sim),
        'stratified_se': np.sqrt(strat_rate * (1 - strat_rate) / n_sim),
        'mcnemar_var_mean':   np.mean(mcn_vars)   if mcn_vars   else np.nan,
        'stratified_var_mean': np.mean(strat_vars) if strat_vars else np.nan,
        'var_ratio_mean': np.mean(var_ratios) if var_ratios else np.nan,
        'var_ratio_std':  np.std(var_ratios)  if var_ratios else np.nan,
        'power_diff': strat_rate - mcn_rate,
        'n_sim': n_sim
    }

# =============================================================================
# Configuration
# =============================================================================

print(f"\n{'='*70}")
print("CONFIGURATION")
print(f"{'='*70}")

N_SIM      = 50000
ALPHA      = 0.05
BASE_SEED  = 42

print(f"n_sim per scenario : {N_SIM:,}")
print(f"alpha              : {ALPHA}")
print(f"base_seed          : {BASE_SEED}")

# =============================================================================
# DOE Scenarios
# =============================================================================

print(f"\n{'='*70}")
print("DOE SCENARIOS")
print(f"{'='*70}")

all_scenarios = []

# ---------- shared N grids ----------
N_EXTENDED = [30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
N_STANDARD = [50, 100, 200, 500, 1000, 2000, 5000]
N_SMALL    = [30, 50, 75, 100]
N_POWER    = [50, 100, 200, 500, 1000, 2000]

AB_SQ_LEVELS      = [0.00, 0.01, 0.04, 0.09, 0.16, 0.25]
RATIOS            = [1.0, 0.5, 0.3, 0.2, 0.1]
DELTA_ACC_LEVELS  = [0.05, 0.10, 0.15, 0.20]

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Heterogeneity Levels – fixed Disc = 0.5
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Part 1] Heterogeneity Levels (Disc = 0.5)")

for N in N_STANDARD:
    for ab_sq in AB_SQ_LEVELS:
        a, b = ab_sq_to_ab(ab_sq)
        if abs(a) <= 0.5:
            all_scenarios.append({
                'part': 1, 'part_name': 'Hetero_Disc0.5',
                'N': N, 'ratio': 1.0,
                'disc_pos': 0.5, 'disc_neg': 0.5,
                'a': a, 'b': b,
                'effect_type': f'ab_sq_{ab_sq}',
                'ab_squared': ab_sq, 'delta_acc_true': 0.0,
                'expected_disc_count': np.nan
            })

print(f"  Part 1 scenarios: {len([s for s in all_scenarios if s['part']==1])}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Discordant Rate × Heterogeneity
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Part 2] Disc Rate × Heterogeneity")

PART2_AB_SQ = [0.00, 0.01, 0.04, 0.09, 0.16, 0.25]
PART2_DISC  = [0.1, 0.2, 0.3, 0.4, 0.5]

for N in N_STANDARD:
    for ab_sq in PART2_AB_SQ:
        a, b = ab_sq_to_ab(ab_sq)
        for disc in PART2_DISC:
            if abs(a) <= disc:
                all_scenarios.append({
                    'part': 2, 'part_name': 'Disc_x_Hetero',
                    'N': N, 'ratio': 1.0,
                    'disc_pos': disc, 'disc_neg': disc,
                    'a': a, 'b': b,
                    'effect_type': f'disc_{disc}_ab_{ab_sq}',
                    'ab_squared': ab_sq, 'delta_acc_true': 0.0,
                    'expected_disc_count': np.nan
                })

print(f"  Part 2 scenarios: {len([s for s in all_scenarios if s['part']==2])}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Power – Heterogeneous
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Part 3] Power – Heterogeneous")

def get_ab_for_delta_and_hetero(delta_acc, ab_sq):
    """
    Given ΔAcc and (a-b)², compute (a,b) for balanced classes.
    a + b = 2*ΔAcc  and  a - b = sqrt(ab_sq)
    => a = ΔAcc + sqrt(ab_sq)/2,  b = ΔAcc - sqrt(ab_sq)/2
    """
    ab_diff = np.sqrt(ab_sq)
    a = delta_acc + ab_diff / 2
    b = delta_acc - ab_diff / 2
    return round(a, 4), round(b, 4)

for N in N_POWER:
    for ab_sq in AB_SQ_LEVELS:
        for delta_acc in DELTA_ACC_LEVELS:
            a, b = get_ab_for_delta_and_hetero(delta_acc, ab_sq)
            if abs(a) <= 0.5 and abs(b) <= 0.5:
                all_scenarios.append({
                    'part': 3, 'part_name': 'Power_Hetero',
                    'N': N, 'ratio': 1.0,
                    'disc_pos': 0.5, 'disc_neg': 0.5,
                    'a': a, 'b': b,
                    'effect_type': f'ab_{ab_sq}_dAcc_{delta_acc}',
                    'ab_squared': ab_sq, 'delta_acc_true': delta_acc,
                    'expected_disc_count': np.nan
                })

print(f"  Part 3 scenarios: {len([s for s in all_scenarios if s['part']==3])}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Expected Discordant Count × Heterogeneity  
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Part 4] Expected Discordant Count × Heterogeneity")

PART_4_N     = [30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]
PART_4_DISC  = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  
PART_4_AB_SQ = [0.00, 0.01, 0.04, 0.09, 0.16, 0.25]

for ab_sq in PART_4_AB_SQ:
    a, b = ab_sq_to_ab(ab_sq)
    for N in PART_4_N:
        for disc in PART_4_DISC:
            # abs(a) <= disc is guaranteed by design but keep check for safety
            if abs(a) <= disc:
                expected_disc = (N / 2) * disc   # N_pos = N/2 (ratio = 1.0)
                all_scenarios.append({
                    'part': 4, 'part_name': 'Expected_Disc_Count',
                    'N': N, 'ratio': 1.0,
                    'disc_pos': disc, 'disc_neg': disc,
                    'a': a, 'b': b,
                    'effect_type': f'ab_{ab_sq}_disc_{disc}',
                    'ab_squared': ab_sq, 'delta_acc_true': 0.0,
                    'expected_disc_count': expected_disc
                })

print(f"  Part 4 scenarios: {len([s for s in all_scenarios if s['part']==4])}")

ab_counts = collections.Counter(
    s['ab_squared'] for s in all_scenarios if s['part']==4
)
print(f"  Scenarios per ab_sq panel: {dict(sorted(ab_counts.items()))}")
print(f"  Expected disc range: [{min(PART_4_N)/2*min(PART_4_DISC):.1f}, "
      f"{max(PART_4_N)/2*max(PART_4_DISC):.1f}] ")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("DOE SUMMARY")
print(f"{'='*70}")

scenario_df = pd.DataFrame(all_scenarios)
summary = scenario_df.groupby(['part', 'part_name']).size().reset_index(name='count')
print(summary.to_string(index=False))

total_scenarios = len(all_scenarios)
total_sims      = total_scenarios * N_SIM
print(f"\nTotal scenarios   : {total_scenarios}")
print(f"Total simulations : {total_sims:,}")
print(f"Estimated time    : {total_sims/1e6*2:.0f} – {total_sims/1e6*5:.0f} minutes")

# =============================================================================
# Run All Simulations
# =============================================================================

print(f"\n{'='*70}")
print("RUNNING SIMULATIONS")
print(f"{'='*70}")

start_time = time.time()
results    = []

for i, scenario in enumerate(tqdm(all_scenarios, desc="Progress")):

    result = run_simulation_scenario(
        N        = scenario['N'],
        ratio    = scenario['ratio'],
        disc_pos = scenario['disc_pos'],
        disc_neg = scenario['disc_neg'],
        a        = scenario['a'],
        b        = scenario['b'],
        n_sim    = N_SIM,
        alpha    = ALPHA,
        seed     = BASE_SEED + i,
    )
    if result is not None:
        result['part']                = scenario['part']
        result['part_name']           = scenario['part_name']
        result['effect_type']         = scenario['effect_type']
        result['ab_squared']          = scenario['ab_squared']
        result['delta_acc_true']      = scenario['delta_acc_true']
        result['expected_disc_count'] = scenario['expected_disc_count']
        results.append(result)

results_df   = pd.DataFrame(results)
elapsed_time = time.time() - start_time

print(f"\nSimulation complete!  Time: {elapsed_time/60:.1f} minutes")
print(f"Valid scenarios: {len(results_df)}")

results_df.to_csv('PRIM_accuracy_results.csv', index=False)
