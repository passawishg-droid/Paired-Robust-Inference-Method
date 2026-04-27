"""
BA Test — Figure Plotting Script
=================================
Reads: PRIM_ba_results.csv  (output of prim_balanced_accuracy.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.interpolate import PchipInterpolator as _Pchip
from scipy import stats
from scipy.optimize import brentq
from scipy.stats    import t as t_dist
import matplotlib.ticker as ticker
from numpy.linalg   import inv as np_inv
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import brentq
from scipy.stats    import t as t_dist
from numpy.linalg   import inv as np_inv

# =============================================================================
# CONFIGURATION  (must match the simulation that produced the CSV)
# =============================================================================
N_SIM  = 50000
ALPHA  = 0.05

# =============================================================================
# LOAD DATA
# =============================================================================
import os
CSV_PATH = 'PRIM_ba_results.csv'

results_df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(results_df):,} scenarios from '{CSV_PATH}'")
print(f"Parts present: {sorted(results_df['part'].unique())}")

se_nominal = np.sqrt(ALPHA * (1 - ALPHA) / N_SIM)

# GLOBAL STYLE SETTINGS
# =============================================================================
plt.rcParams.update({
    'font.family'      : 'serif',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : False,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
})

CB_COLORS  = ['#000000','#E69F00','#56B4E9','#009E73',
              '#F0E442','#0072B2','#D55E00','#CC79A7']
CB_MARKERS = ['o','s','^','D','v','P','X','*']

XTICKS_STD = [50, 100, 200, 500, 1000, 2000, 5000]
XLABS_STD  = ['50','100','200','500','1,000','2,000','5,000']

from scipy.interpolate import PchipInterpolator as _Pchip

def ab_sq_to_ab(ab_sq):
    a = round(np.sqrt(ab_sq) / 2, 6)
    return a, -a

# =============================================================================
# FIGURE 1: Type I Error by N+/N- Ratio (fig. 7 in the manuscript)
# =============================================================================
print("\nFigure 1: Type I Error by N+/N- Ratio")

part1   = results_df[results_df['part'] == 1].copy()
ratios1 = sorted(part1['ratio'].unique(), reverse=True)

fig2, ax = plt.subplots(figsize=(9, 6))
for idx, ratio in enumerate(ratios1):
    col = CB_COLORS[idx % len(CB_COLORS)]
    mk  = CB_MARKERS[idx % len(CB_MARKERS)]
    sub = part1[part1['ratio'] == ratio].sort_values('N')
    ax.errorbar(sub['N'], sub['reject_rate'],
                yerr=1.96*sub['reject_se'],
                fmt=f'{mk}-', color=col,
                label=f'$N^+/N^- = {ratio}$',
                markersize=7, capsize=2, linewidth=1.8)

ax.axhline(y=ALPHA, color='#555555', linestyle=':', linewidth=1.5)
ax.axhspan(ALPHA - 1.96*se_nominal, ALPHA + 1.96*se_nominal,
           alpha=0.18, color='#888888', label='95% CI')
ax.set_xlabel('Total Sample Size (N)', fontsize=12)
ax.set_ylabel('Type I Error Rate', fontsize=12)
ax.legend(fontsize=10, loc='upper right', frameon=True,
          framealpha=0.92, edgecolor='#cccccc',
          markerscale=1.3, handlelength=2.0,
          borderpad=0.8, labelspacing=0.5)
ax.set_xscale('log')
ax.set_xticks(XTICKS_STD)
ax.set_xticklabels(XLABS_STD, fontsize=10)
ax.set_ylim([0.03, 0.14])
ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
plt.savefig('ba_fig1_ratio_type1.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# FIGURE 2: Heatmap Type I Error (fig. 8 in the manuscript)
# =============================================================================

print("\nFigure 2: Heatmap Type I Error")
 
part2 = results_df[results_df['part'] == 1].copy()
ba_pivot = part2.pivot_table(values='reject_rate', index='ratio', columns='N')
ba_pivot = ba_pivot.sort_index(ascending=True)  
 
vmin9, vcenter9, vmax9 = 0.030, 0.050, 0.070
norm9 = TwoSlopeNorm(vmin=vmin9, vcenter=vcenter9, vmax=vmax9)
cmap9 = LinearSegmentedColormap.from_list("sym_type1_ba", [
    (0.00, "#8b0000"),
    (0.14, "#d94f00"),
    (0.32, "#f0c030"),
    (0.50, "#2e7d32"),
    (0.68, "#f0c030"),
    (0.86, "#d94f00"),
    (1.00, "#8b0000"),
], N=512)

fig2 = plt.figure(figsize=(9.5, 5.8))
ax9  = fig2.add_axes([0.10, 0.13, 0.70, 0.72])
cax9 = fig2.add_axes([0.85, 0.13, 0.030, 0.72])
 
def draw_heatmap9_ba(ax, pivot, clamp=True):
    from matplotlib.patches import Rectangle as _Rect
    plot_data = np.clip(pivot.values, vmin9, vmax9) if clamp else pivot.values
    nrows, ncols = plot_data.shape
    im = ax.imshow(plot_data, cmap=cmap9, norm=norm9,
                   aspect='auto', interpolation='nearest', origin='upper')
 
    for i in range(nrows):
        for j in range(ncols):
            raw    = pivot.values[i, j]
            normed = norm9(min(raw, vmax9))
            ax.text(j, i, f'{raw:.3f}',
                    ha='center', va='center',
                    fontsize=12, fontfamily='monospace',
                    fontweight='bold' ,
                    color='white' )
 
    for i in range(nrows):
        for j in range(ncols):
            if abs(pivot.values[i, j] - vcenter9) <= 0.005:
                ax.add_patch(_Rect(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    linewidth=1.6, edgecolor='#002200',
                    facecolor='none', zorder=3))

    for x in np.arange(-0.5, ncols, 1):
        ax.axvline(x, color='white', linewidth=0.6, zorder=2)
    for y in np.arange(-0.5, nrows, 1):
        ax.axhline(y, color='white', linewidth=0.6, zorder=2)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels([f'{int(c):,}' for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([f'{x:.1f}' for x in pivot.index], fontsize=10)
    ax.set_ylabel(r'$N^+/N^-$ Ratio', fontsize=11, labelpad=7)
    ax.set_xlabel('Sample Size (N)', fontsize=11, labelpad=7)
    ax.set_title('BA Test', fontsize=13, fontweight='bold', pad=8)
    ax.tick_params(length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return im
 
im9 = draw_heatmap9_ba(ax9, ba_pivot, clamp=True)
 
cbar9 = fig2.colorbar(im9, cax=cax9)
cbar9.set_label('Type I Error', fontsize=10, labelpad=9)
tick_vals9 = [0.030, 0.035, 0.040, 0.045, 0.050,
              0.055, 0.060, 0.065, 0.070]
cbar9.set_ticks(tick_vals9)
cbar9.set_ticklabels([
    '0.050 (\u03b1)' if v == 0.050 else
    '>= 0.070'       if v == vmax9 else
    f'{v:.3f}'       for v in tick_vals9
])
cbar9.ax.tick_params(labelsize=8.5)
 
fig2.suptitle(
    r'Type I Error Heatmap  (Dark Green = $\alpha$ = 0.05,  Red = Deviation)',
    fontsize=13, fontweight='bold', y=0.975
)
fig2.text(0.48, 0.018,
          'Dark-bordered cells: Type I error within \u03b1 \u00b1 0.005.',
          ha='center', va='bottom', fontsize=9,
          color='#444444', style='italic')
 
plt.savefig('ba_fig2_heatmap_type1.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.show()

# =============================================================================
# FIGURE 3: Variance Bias by ratio (fig. 9 in the manuscript)
# =============================================================================
print("\nFigure 3: Variance Bias (True / Estimated)")

def true_ba_var(N, ratio, disc_pos, disc_neg, a, b):
    N_pos = max(int(round(N * ratio / (1 + ratio))), 1)
    N_neg = max(N - N_pos, 1)
    var_R = (disc_pos - a**2) / N_pos
    var_S = (disc_neg - b**2) / N_neg
    return 0.25 * (var_R + var_S)

part3 = results_df[results_df['part'] == 1].copy()
part3['var_true'] = part3.apply(
    lambda r: true_ba_var(r['N'], r['ratio'],
                          r['disc_pos'], r['disc_neg'],
                          r['a'], r['b']), axis=1)
part3['var_bias_ratio'] = part3['var_true'] / part3['var_BA_mean']

ratios3 = sorted(part3['ratio'].unique(), reverse=True)

fig3, ax = plt.subplots(figsize=(9, 6))
for idx, ratio in enumerate(ratios3):
    col   = CB_COLORS[idx % len(CB_COLORS)]
    mk    = CB_MARKERS[idx % len(CB_MARKERS)]
    sub   = part3[part3['ratio'] == ratio].sort_values('N')
    x_pts = sub['N'].values
    y_pts = sub['var_bias_ratio'].values
    x_sm  = np.exp(np.linspace(np.log(x_pts[0]), np.log(x_pts[-1]), 300))
    y_sm  = _Pchip(np.log(x_pts), y_pts)(np.log(x_sm))
    ax.plot(x_sm, y_sm, color=col, linewidth=1.8, alpha=0.85)
    ax.plot(x_pts, y_pts, mk, color=col, markersize=7,
            label=f'$N^+/N^- = {ratio}$')

ax.axhline(y=1.0, color='#444444', linestyle=':', linewidth=1.8,
           label='Unbiased (= 1)')
ax.set_xlabel('Total Sample Size (N)', fontsize=12)
ax.set_ylabel(r'Variance Ratio (Var$_\mathrm{true}$ / Var$_\mathrm{est}$)', fontsize=12)

ax.legend(fontsize=10, loc='upper right', frameon=True,
          framealpha=0.92, edgecolor='#cccccc',
          markerscale=1.3, handlelength=2.0,
          borderpad=0.8, labelspacing=0.5)
ax.set_xscale('log')
ax.set_xticks(XTICKS_STD)
ax.set_xticklabels(XLABS_STD, fontsize=10)
ax.set_ylim([0.99, 1.25])
ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
plt.savefig('ba_fig3_var_bias.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# FIGURE 4: Power by ratio (fig. 10(a-e) in the manuscript)
# =============================================================================
print("\nFigure 4: Power")

part4       = results_df[results_df['part'] == 3].copy()
ratios4     = sorted(part4['ratio'].unique(), reverse=True)
delta_vals4 = sorted(part4['delta_BA_true'].unique())
LEGEND_PANEL_7_BA = 0.5

for idx, ratio in enumerate(ratios4):
    fig7_i, ax = plt.subplots(figsize=(7, 5.5))
    sub = part4[part4['ratio'] == ratio]
    show_leg = (ratio == LEGEND_PANEL_7_BA)

    for jdx, delta in enumerate(delta_vals4):
        col = CB_COLORS[jdx % len(CB_COLORS)]
        mk  = CB_MARKERS[jdx % len(CB_MARKERS)]
        ds  = sub[sub['delta_BA_true'] == delta].sort_values('N')
        if len(ds) < 2:
            continue
        x_pts = ds['N'].values
        y_pts = ds['reject_rate'].values
        x_sm  = np.exp(np.linspace(np.log(x_pts[0]), np.log(x_pts[-1]), 300))
        y_sm  = _Pchip(np.log(x_pts), y_pts)(np.log(x_sm))
        lbl   = f'$\\Delta$BA = {delta:.2f}' if show_leg else None
        ax.plot(x_sm, y_sm, '-', color=col, linewidth=2.0)
        ax.plot(x_pts, y_pts, mk, color=col, markersize=6, label=lbl)

    ax.set_xlabel('Total Sample Size (N)', fontsize=12)
    ax.set_ylabel('Power (Rejection Rate)', fontsize=12)
    ax.set_title(f'$N^+/N^- = {ratio}$', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(XTICKS_STD)
    ax.set_xticklabels(XLABS_STD, fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.tick_params(axis='y', labelsize=10)

    if show_leg:
        leg = ax.legend(fontsize=12, loc='lower right',
                        frameon=True, framealpha=0.92, edgecolor='#cccccc',
                        markerscale=1.6, handlelength=2.5,
                        borderpad=0.9, labelspacing=0.6)
        leg._legend_box.align = 'left'

    plt.tight_layout()
    fname = f'ba_fig4_power_ratio{str(ratio).replace(".", "")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {fname}")


# =============================================================================
# FIGURE B: Minimum Expected Discordant Count (BA Test) (does not appear in the manuscript)
# =============================================================================

print("\nFigure B: Minimum Expected Discordant Count (BA Test)")


def ols_fit_with_ci(x_fit, y_fit, x_smooth, ci_level=0.95):
    n       = len(x_fit)
    logx    = np.log(x_fit)
    X       = np.column_stack([np.ones(n), logx, logx**2])
    XtX_inv = np_inv(X.T @ X)
    beta    = XtX_inv @ (X.T @ y_fit)
    resid   = y_fit - X @ beta
    s2      = (resid @ resid) / (n - 3)
    logx_s  = np.log(x_smooth)
    X_s     = np.column_stack([np.ones(len(x_smooth)), logx_s, logx_s**2])
    y_hat   = X_s @ beta
    se_fit  = np.sqrt(s2 * np.einsum('ij,jk,ik->i', X_s, XtX_inv, X_s))
    t_crit  = t_dist.ppf(1 - (1 - ci_level) / 2, df=n - 3)
    return y_hat, y_hat - t_crit * se_fit, y_hat + t_crit * se_fit


def find_crossing(x_smooth, y_curve, level):
    diff         = y_curve - level
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return x_smooth[0] if y_curve[0] <= level else np.nan
    i = sign_changes[0]
    try:
        log_cross = brentq(
            lambda lx: np.interp(lx, np.log(x_smooth), y_curve) - level,
            np.log(x_smooth[i]), np.log(x_smooth[i + 1])
        )
        return np.exp(log_cross)
    except ValueError:
        return np.nan


MIN_DISC_FOR_FIT = 5
TREND_CI_LEVEL   = 0.95

part_b   = results_df[results_df['part'] == 2].copy()
ab_sq    = 0.00  # single panel — homogeneous only

subset     = part_b.sort_values('expected_disc_count')
fit_subset = subset[subset['expected_disc_count'] >= MIN_DISC_FOR_FIT]

x_fit    = fit_subset['expected_disc_count'].values
y_str    = fit_subset['reject_rate'].values
x_smooth = np.exp(np.linspace(np.log(MIN_DISC_FOR_FIT),
                               np.log(subset['expected_disc_count'].max()), 400))

y_s, lo_s, hi_s = ols_fit_with_ci(x_fit, y_str, x_smooth, ci_level=TREND_CI_LEVEL)
upper_band = ALPHA + 1.96 * se_nominal
threshold  = find_crossing(x_smooth, y_s, upper_band)

print(f"\n{'='*65}")
print("BA TEST — MINIMUM EXPECTED DISCORDANT COUNT PER CLASS")
print(f"Criterion: OLS trend <= upper_band (α + 1.96·SE = {upper_band:.4f})")
print(f"Threshold: {threshold:.1f} expected discordant pairs per class")
print(f"{'='*65}")

fig_bi, ax = plt.subplots(figsize=(9, 6))

excl = subset[subset['expected_disc_count'] < MIN_DISC_FOR_FIT]
incl = subset[subset['expected_disc_count'] >= MIN_DISC_FOR_FIT]

ax.scatter(excl['expected_disc_count'], excl['reject_rate'],
           marker='o', color='darkblue', alpha=0.15, s=25, zorder=2)
ax.scatter(incl['expected_disc_count'], incl['reject_rate'],
           marker='o', color='darkblue', alpha=0.55, s=35,
           label='BA Stratified', zorder=3)

ax.plot(x_smooth, y_s, '-', color='darkblue', linewidth=2, alpha=0.85, label='OLS Trend')
ax.fill_between(x_smooth, lo_s, hi_s, color='darkblue', alpha=0.12,
                label=f'Trend {int(TREND_CI_LEVEL*100)}% CI')

ax.axhline(y=ALPHA, color='gray', linestyle=':', linewidth=2, label=f'α={ALPHA}')
ax.axhspan(ALPHA - 1.96*se_nominal, ALPHA + 1.96*se_nominal,
           alpha=0.22, color='gray', label='95% accept band')

if not np.isnan(threshold):
    ax.axvline(x=threshold, color='darkgreen', linestyle='-.', linewidth=2,
               alpha=0.9, label=f'Threshold: {threshold:.1f}')
    ax.text(threshold, 0.0265, f'{threshold:.1f}',
            color='darkgreen', fontsize=9, ha='center', fontweight='bold', va='bottom')

ax.axvline(x=MIN_DISC_FOR_FIT, color='black', linestyle=':',
           linewidth=1, alpha=0.35, label=f'Fit start ({MIN_DISC_FOR_FIT})')

ax.set_xlabel('Expected Discordant Count per Class  (N/2 × Disc)', fontsize=11)
ax.set_ylabel('Type I Error Rate', fontsize=11)
ax.set_title(f'Figure B: BA Test — Minimum Expected Discordant Count\n'
             f'Fixed: ratio=1.0, a=b=0 (H₀)  |  n_sim={N_SIM:,}  |  '
             f'OLS ± {int(TREND_CI_LEVEL*100)}% CI',
             fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.set_ylim([0.025, 0.085])
ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
plt.tight_layout()
plt.savefig('ba_figB_expected_disc.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: ba_figB_expected_disc.png")
