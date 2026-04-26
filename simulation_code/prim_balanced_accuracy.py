"""
PRI Method Simulation Study (Balanced Accuracy (BA) Test)
"""

# =============================================================================
# Setup
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.stats    import t as t_dist
from numpy.linalg   import inv as np_inv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
import warnings
import time
import collections
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("PRIM SIMULATION STUDY — BA TEST")
print("Stratified Variance for ΔBA  |  H0: ΔBA = 0  (a + b = 0)")
print("="*70)

# =============================================================================
# Core Utility Functions
# =============================================================================

def generate_contingency_table(n, p10, p01, p11=None):
    """Generate 2×2 contingency table from multinomial."""
    assert 0 <= p10 <= 1 and 0 <= p01 <= 1
    assert p10 + p01 <= 1
    if p11 is None:
        remaining = 1 - p10 - p01
        p11 = remaining * 0.6
        p00 = remaining * 0.4
    else:
        p00 = 1 - p11 - p10 - p01
        assert p00 >= 0
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
        'n11_pos': n11_pos, 'n10_pos': n10_pos, 'n01_pos': n01_pos, 'n00_pos': n00_pos,
        'n11_neg': n11_neg, 'n10_neg': n10_neg, 'n01_neg': n01_neg, 'n00_neg': n00_neg,
        'n11': n11_pos+n11_neg, 'n10': n10_pos+n10_neg,
        'n01': n01_pos+n01_neg, 'n00': n00_pos+n00_neg,
    }


def params_from_effect(disc_rate, effect):
    """Convert (disc_rate, effect) → (p10, p01)."""
    p10 = (disc_rate + effect) / 2
    p01 = (disc_rate - effect) / 2
    if p10 < 0 or p01 < 0 or p10 > 1 or p01 > 1:
        raise ValueError(f"Invalid: disc={disc_rate}, effect={effect} "
                         f"→ p10={p10:.4f}, p01={p01:.4f}")
    return p10, p01


def ab_sq_to_ab(ab_sq):
    """Convert (a-b)² → (a, b) symmetric: a = √(ab_sq)/2, b = -a."""
    a = round(np.sqrt(ab_sq) / 2, 6)
    return a, -a


def ba_stratified_test(data):
    """
    Stratified test for H0: ΔBA = 0.
    BA   = (Recall + Specificity) / 2
    ΔBA  = (ΔR + ΔS) / 2   where ΔR = a, ΔS = b
    Var(ΔBA) = (1/4)[Var(ΔR) + Var(ΔS)]   (independence of disjoint strata)
    """
    N_pos, N_neg   = data['N_pos'], data['N_neg']
    n10_pos, n01_pos = data['n10_pos'], data['n01_pos']
    n10_neg, n01_neg = data['n10_neg'], data['n01_neg']

    delta_R  = (n10_pos - n01_pos) / N_pos if N_pos > 0 else 0.0
    delta_S  = (n10_neg - n01_neg) / N_neg if N_neg > 0 else 0.0
    delta_BA = (delta_R + delta_S) / 2

    var_R = max((n10_pos+n01_pos)/N_pos**2 - (n10_pos-n01_pos)**2/N_pos**3, 0) \
            if N_pos > 0 and (n10_pos+n01_pos) > 0 else 0.0
    var_S = max((n10_neg+n01_neg)/N_neg**2 - (n10_neg-n01_neg)**2/N_neg**3, 0) \
            if N_neg > 0 and (n10_neg+n01_neg) > 0 else 0.0

    var_BA = (1/4) * (var_R + var_S)

    if var_BA <= 0:
        return {'delta_BA': delta_BA, 'delta_R': delta_R, 'delta_S': delta_S,
                'var_R': var_R, 'var_S': var_S, 'var_BA': np.nan,
                'se_BA': np.nan, 'z': 0.0, 'p_value': 1.0}

    se_BA   = np.sqrt(var_BA)
    z       = delta_BA / se_BA
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {'delta_BA': delta_BA, 'delta_R': delta_R, 'delta_S': delta_S,
            'var_R': var_R, 'var_S': var_S, 'var_BA': var_BA,
            'se_BA': se_BA, 'z': z, 'p_value': p_value}


def run_simulation_scenario(N, ratio, disc_pos, disc_neg, a, b,
                            n_sim=10000, alpha=0.05, seed=None,
                            p11_pos=None, p11_neg=None):
    """Run simulation for a single scenario and return aggregated results. """
    if seed is not None:
        np.random.seed(seed)

    N_pos = max(int(round(N * ratio / (1 + ratio))), 1)
    N_neg = max(N - N_pos, 1)

    try:
        p10_pos, p01_pos = params_from_effect(disc_pos, a)
        p10_neg, p01_neg = params_from_effect(disc_neg, b)
    except ValueError as e:
        print(f"  Warning: {e}")
        return None

    rejects        = 0
    z_values       = []
    var_estimates  = []
    delta_BA_vals  = []

    for _ in range(n_sim):
        data   = generate_stratified_data(N_pos, N_neg,
                                          p10_pos, p01_pos,
                                          p10_neg, p01_neg,
                                          p11_pos=p11_pos,
                                          p11_neg=p11_neg)
        result = ba_stratified_test(data)
        if result['p_value'] < alpha:
            rejects += 1
        z_values.append(result['z'])
        delta_BA_vals.append(result['delta_BA'])
        if not np.isnan(result['var_BA']) and result['var_BA'] > 0:
            var_estimates.append(result['var_BA'])

    reject_rate = rejects / n_sim

    return {
        'N': N, 'N_pos': N_pos, 'N_neg': N_neg, 'ratio': ratio,
        'disc_pos': disc_pos, 'disc_neg': disc_neg,
        'a': a, 'b': b,
        'ab_squared'    : round((a - b)**2, 8),
        'delta_BA_true' : (a + b) / 2,
        'reject_rate'   : reject_rate,
        'reject_se'     : np.sqrt(reject_rate * (1 - reject_rate) / n_sim),
        'z_mean'        : np.mean(z_values),
        'z_std'         : np.std(z_values),
        'var_BA_mean'   : np.mean(var_estimates) if var_estimates else np.nan,
        'delta_BA_mean' : np.mean(delta_BA_vals),
        'delta_BA_std'  : np.std(delta_BA_vals),
        'n_sim'         : n_sim,
    }


# =============================================================================
# Configuration
# =============================================================================
print(f"\n{'='*70}")
print("CONFIGURATION")
print(f"{'='*70}")

N_SIM     = 50000
ALPHA     = 0.05
BASE_SEED = 42

print(f"n_sim per scenario : {N_SIM:,}")
print(f"alpha              : {ALPHA}")
print(f"base_seed          : {BASE_SEED}")

# =============================================================================
# STEP 4: DOE Scenarios
# =============================================================================
print(f"\n{'='*70}")
print("DOE SCENARIOS")
print(f"{'='*70}")

all_scenarios = []

# ─────────────────────────────────────────────────────────────────────────────
# Type I Error — Fixed N=500, vary N+/N- ratio
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Part 1] Type I Error — Fixed N, vary ratio")

PART1_N      = [50, 100, 200, 500, 1000, 2000, 5000]
PART1_RATIOS = [1.0, 0.5, 0.3, 0.2, 0.1]
PART1_DISC   = 0.5

for N in PART1_N:
    for ratio in PART1_RATIOS:
        all_scenarios.append({
            'part': 1, 'part_name': 'Ratio_Type1', 'type': 'Type1',
            'N': N, 'ratio': ratio,
            'disc_pos': PART1_DISC, 'disc_neg': PART1_DISC,
            'a': 0.0, 'b': 0.0,
            'ab_squared': 0.0,
            'effect_type': f'ratio={ratio}',
            'expected_disc_count': np.nan,
        })

print(f"  Part 1 scenarios: {len([s for s in all_scenarios if s['part']==1])}")

# ─────────────────────────────────────────────────────────────────────────────
# Minimum Expected Discordant Count
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Part 2] Expected Discordant Count (BA Test)")

PART_2_N      = [30, 40, 50, 60, 75, 100, 125, 150, 175, 200,
                 250, 300, 400, 500]
PART_2_DISC   = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
PART_2_AB_SQ  = [0.00]   

for ab_sq in PART_2_AB_SQ:
    a, b = ab_sq_to_ab(ab_sq)
    for N in PART_2_N:
        for disc in PART_2_DISC:
            expected_disc = (N / 2) * disc
            all_scenarios.append({
                'part': 2, 'part_name': 'Expected_Disc_Count', 'type': 'Type1',
                'N': N, 'ratio': 1.0,
                'disc_pos': disc, 'disc_neg': disc,
                'a': a, 'b': b,
                'ab_squared': ab_sq,
                'effect_type': f'disc_{disc}',
                'expected_disc_count': expected_disc,
            })

print(f"  Part 2 scenarios: {len([s for s in all_scenarios if s['part']==2])}")
ab_counts = collections.Counter(s['ab_squared'] for s in all_scenarios if s['part']==2)
print(f"  Scenarios per ab_sq: {dict(sorted(ab_counts.items()))}")
print(f"  Expected disc range: [{min(PART_2_N)/2*min(PART_2_DISC):.1f}, "
      f"{max(PART_2_N)/2*max(PART_2_DISC):.1f}] (identical for all panels)")

for s in all_scenarios:
    if 'N_minority'  not in s: s['N_minority']   = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Power — Fixed N=500, vary ratio × ΔBA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Part 3] Power — Fixed N=500, vary ratio × ΔBA")

PART3_N        = [50, 100, 200, 500, 1000, 2000, 5000]
PART3_RATIOS   = [1.0, 0.5, 0.3, 0.2, 0.1]
PART3_DELTA_BA = [0.05, 0.10, 0.15, 0.20]

for N in PART3_N:
    for ratio in PART3_RATIOS:
        for delta_BA in PART3_DELTA_BA:
            a = delta_BA
            b = delta_BA
            all_scenarios.append({
                'part': 3, 'part_name': 'Power_Ratio', 'type': 'Power',
                'N': N, 'ratio': ratio,
                'disc_pos': 0.5, 'disc_neg': 0.5,
                'a': a, 'b': b,
                'ab_squared': 0.0,
                'delta_BA_true': delta_BA,
                'effect_type': f'ratio={ratio}_dBA={delta_BA:.2f}',
                'expected_disc_count': np.nan,
            })

print(f"  Part 3 scenarios: {len([s for s in all_scenarios if s['part']==3])}")

# ── DOE Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("DOE SUMMARY")
print(f"{'='*70}")
scenario_df = pd.DataFrame(all_scenarios)
summary = scenario_df.groupby(['part','part_name','type']).size().reset_index(name='count')
print(summary.to_string(index=False))
total_scenarios = len(all_scenarios)
print(f"\nTotal scenarios  : {total_scenarios:,}")
print(f"Total simulations: {total_scenarios * N_SIM:,}")

# =============================================================================
# Run Simulations
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
        result['type']                = scenario['type']
        result['effect_type']         = scenario['effect_type']
        result['ab_squared']          = scenario['ab_squared']
        result['expected_disc_count'] = scenario['expected_disc_count']
        result['N_minority']          = scenario.get('N_minority', np.nan)
        result['delta_BA_true']       = scenario.get('delta_BA_true', (scenario['a'] + scenario['b']) / 2)
        results.append(result)

results_df = pd.DataFrame(results)

elapsed = time.time() - start_time
print(f"\nSimulation complete in {elapsed/60:.1f} minutes")
print(f"Valid scenarios: {len(results_df):,}")

results_df.to_csv('PRIM_ba_results.csv', index=False)
