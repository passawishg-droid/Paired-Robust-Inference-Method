"""
PRIM Accuracy Test — Figure Plotting Script
============================================
Reads: PRIM_accuracy_results.csv  (output of prim_accuracy.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.interpolate import PchipInterpolator
from scipy import stats
import warnings
import matplotlib.ticker as ticker
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
CSV_PATH = 'PRIM_accuracy_results.csv'

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
XTICKS_POW = [50, 100, 200, 500, 1000, 2000]
XLABS_POW  = ['50','100','200','500','1,000','2,000']

from scipy.interpolate import PchipInterpolator

# =============================================================================
# FIGURE 1: Disc Rate x Heterogeneity  (fig. 1(a-f) in the manuscript)
# =============================================================================
print("\nFigure 1: Disc Rate x Heterogeneity")

part1        = results_df[results_df['part'] == 2].copy()
ab_sq_groups = sorted(part1['ab_squared'].unique())
LEGEND_PANEL_3 = 0.01

for idx, ab_sq in enumerate(ab_sq_groups):
    fig3_i, ax = plt.subplots(figsize=(7, 5.5))
    subset      = part1[part1['ab_squared'] == ab_sq]
    disc_values = sorted(subset['disc_pos'].unique())
    se_ci       = 1.96 * se_nominal

    for jdx, disc in enumerate(disc_values):
        col = CB_COLORS[jdx % len(CB_COLORS)]
        mk  = CB_MARKERS[jdx % len(CB_MARKERS)]
        ds  = subset[subset['disc_pos'] == disc].sort_values('N')
        lbl = f'disc = {disc}' if ab_sq == LEGEND_PANEL_3 else None

        ax.errorbar(ds['N'] * 1.00, ds['mcnemar_reject_rate'],
                    yerr=se_ci, fmt=f'{mk}--', color=col,
                    alpha=0.55, markersize=5, linewidth=1.3,
                    capsize=2, elinewidth=0.8)

        ax.errorbar(ds['N'] * 1.04, ds['stratified_reject_rate'],
                    yerr=se_ci, fmt=f'{mk}-', color=col,
                    alpha=0.95, markersize=5, linewidth=1.8,
                    capsize=2, elinewidth=0.8, label=lbl)

    ax.axhline(y=ALPHA, color='#555555', linestyle=':', linewidth=1.5)
    ax.axhspan(ALPHA - 1.96*se_nominal, ALPHA + 1.96*se_nominal,
               alpha=0.18, color='#888888',
               label='95% CI' if ab_sq == LEGEND_PANEL_3 else None)

    ax.set_xlabel('Sample Size (N)', fontsize=12)
    ax.set_ylabel('Type I Error Rate', fontsize=12)

    ax.set_title(f'$(a-b)^2 = {ab_sq}$', fontsize=13, fontweight='bold',
                 pad=18)
    ax.set_xscale('log')
    ax.set_xticks(XTICKS_STD)
    ax.set_xticklabels(XLABS_STD, fontsize=10)
    ax.set_ylim([0.02, 0.08])
    ax.tick_params(axis='y', labelsize=10)

    if ab_sq == LEGEND_PANEL_3:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        h_solid  = Line2D([0],[0], color='#555555', linewidth=2.0)
        h_dashed = Line2D([0],[0], color='#555555', linewidth=1.3, linestyle='--')
        h_ci     = Patch(facecolor='#888888', alpha=0.35)

        auto_h, auto_l = ax.get_legend_handles_labels()
        disc_hl = [(h, l) for h, l in zip(auto_h, auto_l)
                   if l is not None and 'disc' in str(l)]

        ordered_h = [h_solid,         h_ci,
                     disc_hl[0][0],   disc_hl[1][0],
                     h_dashed,        disc_hl[2][0],
                     disc_hl[3][0],   disc_hl[4][0]]
        ordered_l = ['Solid = PRI Method', '95% CI',
                     disc_hl[0][1],   disc_hl[1][1],
                     'Dashed = McNemar', disc_hl[2][1],
                     disc_hl[3][1],        disc_hl[4][1]]

        leg = ax.legend(ordered_h, ordered_l,
                        fontsize=11, loc='upper right',
                        frameon=True, framealpha=0.92, edgecolor='#cccccc',
                        ncol=2,
                        markerscale=1.4, handlelength=2.0,
                        borderpad=0.8, labelspacing=0.55,
                        columnspacing=0.8,
                        handletextpad=0.5)
        leg._legend_box.align = 'left'

    plt.tight_layout()
    fname = f'fig1_disc_hetero_absq{str(ab_sq).replace(".", "")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {fname}")

# =============================================================================
# FIGURE 2: Type I Error Heatmap (fig.2(a-b) in the manuscript)
# =============================================================================

print("\nFigure 2: Type I Error Heatmap")
 
part2_heat = results_df[results_df['part'] == 1].copy()
 
mcn_pivot   = part2_heat.pivot_table(
    values='mcnemar_reject_rate',    index='ab_squared', columns='N')
strat_pivot = part2_heat.pivot_table(
    values='stratified_reject_rate', index='ab_squared', columns='N')

vmin9, vcenter9, vmax9 = 0.030, 0.050, 0.070
norm9 = TwoSlopeNorm(vmin=vmin9, vcenter=vcenter9, vmax=vmax9)
cmap9 = LinearSegmentedColormap.from_list("sym_type1", [
    (0.00, "#8b0000"),
    (0.14, "#d94f00"),
    (0.32, "#f0c030"),
    (0.50, "#2e7d32"),
    (0.68, "#f0c030"),
    (0.86, "#d94f00"),
    (1.00, "#8b0000"),
], N=512)

fig2 = plt.figure(figsize=(15, 6))
gs   = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.06)
ax_mcn  = fig2.add_subplot(gs[0])
ax_str  = fig2.add_subplot(gs[1])
ax_cbar = fig2.add_subplot(gs[2])
 
def draw_heatmap9(ax, pivot, title, show_ylabel=True):
    from matplotlib.patches import Rectangle as _Rect
    plot_data = np.clip(pivot.values, vmin9, vmax9)
    nrows, ncols = plot_data.shape
    im = ax.imshow(plot_data, cmap=cmap9, norm=norm9,
                   aspect='auto', interpolation='nearest', origin='upper')
    
    for i in range(nrows):
        for j in range(ncols):
            raw    = pivot.values[i, j]
            normed = norm9(min(raw, vmax9))
            use_white = normed < 0.28 or normed > 0.80
            ax.text(j, i, f'{raw:.3f}',
                    ha='center', va='center',
                    fontsize=11, fontfamily='monospace',
                    fontweight='bold',
                    color='white')
    
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
    if show_ylabel:
        ax.set_yticklabels([f'{x:.2f}' for x in pivot.index], fontsize=10)
        ax.set_ylabel(r'Heterogeneity $(a-b)^2$', fontsize=11, labelpad=7)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')
    ax.set_xlabel('Sample Size (N)', fontsize=11, labelpad=7)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.tick_params(length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return im
 
im1 = draw_heatmap9(ax_mcn, mcn_pivot,   'McNemar Test',    show_ylabel=True)
im2 = draw_heatmap9(ax_str, strat_pivot, 'PRIM Test', show_ylabel=False)

# ── Colorbar ──────────────────────────────────────────────────────────────
cbar9 = fig2.colorbar(im2, cax=ax_cbar)
cbar9.set_label('Type I Error', fontsize=10, labelpad=9)
tick_vals9 = [0.030, 0.035, 0.040, 0.045, 0.050,
              0.055, 0.060, 0.065, 0.070]
cbar9.set_ticks(tick_vals9)
cbar9.set_ticklabels([
    '0.050 (\u03b1)' if v == 0.050 else f'{v:.3f}' for v in tick_vals9
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
 
plt.savefig('fig2_heatmap_type1.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.show()

# =============================================================================
# FIGURE 3: McNemar Variance Ratio (fig.3 in the manuscript)
# =============================================================================
print("\nFigure 3: Variance Ratio")

part3_var   = results_df[results_df['part'] == 1].copy()
ab_sq_values = sorted(part3_var['ab_squared'].unique())

fig5, ax = plt.subplots(figsize=(9, 6))

for idx, ab_sq in enumerate(ab_sq_values):
    col    = CB_COLORS[idx % len(CB_COLORS)]
    mk     = CB_MARKERS[idx % len(CB_MARKERS)]
    subset = part3_var[part3_var['ab_squared'] == ab_sq].sort_values('N')
    x_pts  = subset['N'].values
    y_pts  = subset['var_ratio_mean'].values
    x_sm   = np.exp(np.linspace(np.log(x_pts[0]), np.log(x_pts[-1]), 300))
    y_sm   = PchipInterpolator(np.log(x_pts), y_pts)(np.log(x_sm))
    
    vif_ref = 1.0 / (1.0 - ab_sq / 2.0) if ab_sq < 2.0 else np.nan
    if not np.isnan(vif_ref):
        ax.axhline(y=vif_ref, color=col, linestyle='--', linewidth=0.9,
                   alpha=0.55, zorder=2)
        ax.plot(x_pts, [vif_ref] * len(x_pts), mk, color=col,
                markersize=6, alpha=0.70, zorder=3,
                markeredgewidth=0.4, markeredgecolor='white')

    ax.plot(x_sm, y_sm, color=col, linewidth=1.8, alpha=0.85, zorder=4)
    ax.plot(x_pts, y_pts, mk, color=col, markersize=7, zorder=5,
            label=f'$(a-b)^2 = {ab_sq}$')

ax.set_xlabel('Sample Size (N)', fontsize=12)
ax.set_ylabel('Variance Ratio (Var$_\mathrm{McNemar}$ / Var$_\mathrm{PRIM}$)', fontsize=12)
ax.set_title('Variance Overestimation by Heterogeneity\n'
             r'Dashed lines: theoretical VIF asymptote $(1-(a-b)^2/2)^{-1}$',
             fontsize=13)

from matplotlib.patches import Patch as _Patch
from matplotlib.lines import Line2D as _Line2D
auto_h, auto_l = ax.get_legend_handles_labels()
hetero_hl = [(h, l) for h, l in zip(auto_h, auto_l)]

if len(hetero_hl) % 2 != 0:
    hetero_hl.append((_Patch(visible=False), ''))

h_sol = _Line2D([0],[0], color='#555555', linewidth=1.8)
h_das = _Line2D([0],[0], color='#555555', linewidth=0.9, linestyle='--')

ordered_h = [
    h_sol, hetero_hl[0][0],  
    hetero_hl[1][0], hetero_hl[2][0],
    h_das, hetero_hl[3][0],    
    hetero_hl[4][0], hetero_hl[5][0],  
]
ordered_l = [
    'Solid = Empirical VIF', hetero_hl[0][1],           
    hetero_hl[1][1],hetero_hl[2][1], 
    'Dashed = Theoretical VIF', hetero_hl[3][1],
    hetero_hl[4][1], hetero_hl[5][1],
]

leg = ax.legend(ordered_h, ordered_l,
                fontsize=10, loc='upper right',
                frameon=True, framealpha=0.92, edgecolor='#cccccc',
                ncol=2,
                markerscale=1.3, handlelength=2.0,
                borderpad=0.8, labelspacing=0.5,
                columnspacing=0.8,
                handletextpad=0.5)
leg._legend_box.align = 'left'

ax.set_xscale('log')
ax.set_xticks(XTICKS_STD)
ax.set_xticklabels(XLABS_STD, fontsize=10)
ax.set_ylim([0.99, 1.21])
ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
plt.savefig('fig3_variance_ratio.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# FIGURE 4: PRIM Variance Bias (fig.4 in the manuscript)
# =============================================================================
print("\nFigure 4: Stratified Variance Bias (True / Estimated)")

part4  = results_df[results_df['part'] == 1].copy()
ab_sq_values = sorted(part4['ab_squared'].unique())

def true_stratified_var(N, ratio, disc_pos, disc_neg, a, b):
    N_pos = max(int(round(N * ratio / (1 + ratio))), 1)
    N_neg = max(N - N_pos, 1)
    w_pos = N_pos / N
    w_neg = N_neg / N
    var_R = (disc_pos - a**2) / N_pos
    var_S = (disc_neg - b**2) / N_neg
    return w_pos**2 * var_R + w_neg**2 * var_S

part4['var_true'] = part4.apply(
    lambda r: true_stratified_var(r['N'], r['ratio'],
                                   r['disc_pos'], r['disc_neg'],
                                   r['a'], r['b']), axis=1)
part4['var_bias_ratio'] = part4['var_true'] / part4['stratified_var_mean']

fig5b, ax = plt.subplots(figsize=(9, 4.2))

for idx, ab_sq in enumerate(ab_sq_values):
    col    = CB_COLORS[idx % len(CB_COLORS)]
    mk     = CB_MARKERS[idx % len(CB_MARKERS)]
    subset = part4[part4['ab_squared'] == ab_sq].sort_values('N')
    x_pts  = subset['N'].values
    y_pts  = subset['var_bias_ratio'].values
    x_sm   = np.exp(np.linspace(np.log(x_pts[0]), np.log(x_pts[-1]), 300))
    y_sm   = PchipInterpolator(np.log(x_pts), y_pts)(np.log(x_sm))
    ax.plot(x_sm, y_sm, color=col, linewidth=1.8, alpha=0.85)
    ax.plot(x_pts, y_pts, mk, color=col, markersize=7,
            label=f'$(a-b)^2 = {ab_sq}$')

ax.axhline(y=1.0, color='#444444', linestyle=':', linewidth=1.8,
           label='Unbiased (= 1)')
ax.set_xlabel('Sample Size (N)', fontsize=12)
ax.set_ylabel(r'Variance Ratio (Var$_\mathrm{true}$ / Var$_\mathrm{est}$)', fontsize=12)
ax.set_title('Stratified Estimator Variance Bias\n'
             r'All $(a-b)^2$ lines overlap — bias is finite-sample only',
             fontsize=13)
ax.legend(loc='upper right', fontsize=10, frameon=True,
          framealpha=0.92, edgecolor='#cccccc',
          markerscale=1.3, handlelength=2.0,
          borderpad=0.8, labelspacing=0.5)
ax.set_xscale('log')
ax.set_xticks(XTICKS_STD)
ax.set_xticklabels(XLABS_STD, fontsize=10)
ax.set_ylim([0.995, 1.055])   
ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
plt.savefig('fig5b_stratified_var_bias.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# FIGURE 5: Power Heterogeneous (fig. 5(a-f) in the manuscript)
# =============================================================================
print("\nFigure 5: Power Heterogeneous")

part5       = results_df[results_df['part'] == 3].copy()
ab_sq_values = sorted(part5['ab_squared'].unique())
delta_values = sorted(part5['delta_acc_true'].unique())
LEGEND_PANEL_7 = 0.01

for idx, ab_sq in enumerate(ab_sq_values):
    fig7_i, ax = plt.subplots(figsize=(7, 5.5))
    subset = part5[part5['ab_squared'] == ab_sq]

    for jdx, delta in enumerate(delta_values):
        col = CB_COLORS[jdx % len(CB_COLORS)]
        mk  = CB_MARKERS[jdx % len(CB_MARKERS)]
        ds  = subset[subset['delta_acc_true'] == delta].sort_values('N')
        if len(ds) > 0:
            x_pts = ds['N'].values
            x_sm  = np.exp(np.linspace(np.log(x_pts[0]), np.log(x_pts[-1]), 300))
            y_m   = PchipInterpolator(np.log(x_pts),
                        ds['mcnemar_reject_rate'].values)(np.log(x_sm))
            ax.plot(x_sm, y_m, '--', color=col, alpha=0.55, linewidth=1.5)
            ax.plot(x_pts, ds['mcnemar_reject_rate'].values,
                    mk, color=col, alpha=0.55, markersize=5)
            lbl = f'$\\Delta$Acc = {delta}' if ab_sq == LEGEND_PANEL_7 else None
            y_s = PchipInterpolator(np.log(x_pts),
                        ds['stratified_reject_rate'].values)(np.log(x_sm))
            ax.plot(x_sm, y_s, '-', color=col, linewidth=2.0)
            ax.plot(x_pts, ds['stratified_reject_rate'].values,
                    mk, color=col, markersize=6, label=lbl)

    ax.set_xlabel('Sample Size (N)', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_title(f'$(a-b)^2 = {ab_sq}$', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(XTICKS_POW)
    ax.set_xticklabels(XLABS_POW, fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.tick_params(axis='y', labelsize=10)

    if ab_sq == LEGEND_PANEL_7:
        leg = ax.legend(fontsize=12, loc='lower right',
                        frameon=True, framealpha=0.92, edgecolor='#cccccc',
                        markerscale=1.6, handlelength=2.5,
                        borderpad=0.9, labelspacing=0.6,
                        title='Solid = PRI Method\nDashed = McNemar',
                        title_fontsize=11)
        leg._legend_box.align = 'left'

    plt.tight_layout()
    fname = f'fig5_power_hetero_absq{str(ab_sq).replace(".", "")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {fname}")

# =============================================================================
# FIGURE 6: Expected Discordant Count (fig. 6(a-f) in the manuscript)
# =============================================================================
print("\nFigure 6: Expected Discordant Count x Heterogeneity")

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
            np.log(x_smooth[i]), np.log(x_smooth[i + 1]))
        return np.exp(log_cross)
    except ValueError:
        return np.nan

MIN_DISC_FOR_FIT = 5
TREND_CI_LEVEL   = 0.95
LEGEND_PANEL_B   = 0.01

part_6       = results_df[results_df['part'] == 4].copy()
ab_sq_panels = sorted(part_6['ab_squared'].unique())
fit_store    = {}
thresh_records = []

for ab_sq in ab_sq_panels:
    subset     = part_6[part_6['ab_squared'] == ab_sq].sort_values('expected_disc_count')
    fit_subset = subset[subset['expected_disc_count'] >= MIN_DISC_FOR_FIT]
    if len(fit_subset) < 4:
        fit_store[ab_sq] = None
        continue
    x_fit = fit_subset['expected_disc_count'].values
    y_str = fit_subset['stratified_reject_rate'].values
    y_mcn = fit_subset['mcnemar_reject_rate'].values
    x_smooth = np.exp(np.linspace(np.log(MIN_DISC_FOR_FIT),
                                   np.log(subset['expected_disc_count'].max()), 400))
    y_s, lo_s, hi_s = ols_fit_with_ci(x_fit, y_str, x_smooth, TREND_CI_LEVEL)
    y_m, _,    _    = ols_fit_with_ci(x_fit, y_mcn, x_smooth, TREND_CI_LEVEL)
    fit_store[ab_sq] = {'x_smooth': x_smooth,
                        'y_s': y_s, 'lo_s': lo_s, 'hi_s': hi_s, 'y_m': y_m}
    upper_band = ALPHA + 1.96 * se_nominal
    thresh_records.append({'ab_squared': ab_sq,
                           'point_est' : find_crossing(x_smooth, y_s, upper_band)})

thresh_df = pd.DataFrame(thresh_records)

print(f"\n{'='*55}")
print("MINIMUM EXPECTED DISCORDANT COUNT -- PRIM TEST")
for _, row in thresh_df.iterrows():
    fmt_v = lambda v: f"{v:.1f}" if not np.isnan(v) else "N/A"
    print(f"  (a-b)^2={row['ab_squared']:.2f}  Threshold={fmt_v(row['point_est'])}")

for idx, ab_sq in enumerate(ab_sq_panels):
    fig_bi, ax = plt.subplots(figsize=(7, 5.5))
    subset   = part_6[part_6['ab_squared'] == ab_sq].sort_values('expected_disc_count')
    fit      = fit_store.get(ab_sq)
    show_leg = (ab_sq == LEGEND_PANEL_B)

    COL_STRAT = 'darkblue'   
    COL_MCN   = 'darkred'  
    COL_CUT   = 'black'  

    excl = subset[subset['expected_disc_count'] < MIN_DISC_FOR_FIT]
    incl = subset[subset['expected_disc_count'] >= MIN_DISC_FOR_FIT]

    ax.scatter(excl['expected_disc_count'], excl['stratified_reject_rate'],
               marker='o', color=COL_STRAT, alpha=0.12, s=22, zorder=2)
    ax.scatter(excl['expected_disc_count'], excl['mcnemar_reject_rate'],
               marker='s', color=COL_MCN, alpha=0.12, s=22, zorder=2)
    ax.scatter(incl['expected_disc_count'], incl['stratified_reject_rate'],
               marker='o', color=COL_STRAT, alpha=0.55, s=32, zorder=3,
               label='Stratified' if show_leg else None)
    ax.scatter(incl['expected_disc_count'], incl['mcnemar_reject_rate'],
               marker='s', color=COL_MCN, alpha=0.55, s=32, zorder=3,
               label='McNemar' if show_leg else None)

    if fit is not None:
        xs = fit['x_smooth']
        ax.plot(xs, fit['y_m'], '--', color=COL_MCN, alpha=0.75, linewidth=1.8,
                label='McNemar trend' if show_leg else None)
        ax.plot(xs, fit['y_s'], '-',  color=COL_STRAT, alpha=0.9, linewidth=2.2,
                label='Stratified trend' if show_leg else None)
        ax.fill_between(xs, fit['lo_s'], fit['hi_s'], color=COL_STRAT, alpha=0.12,
                        label='95% CI' if show_leg else None)

    ax.axhline(y=ALPHA, color='#555555', linestyle=':', linewidth=1.5)
    ax.axhspan(ALPHA - 1.96*se_nominal, ALPHA + 1.96*se_nominal,
               alpha=0.15, color='#888888',
               label='95% CI band' if show_leg else None)

    row95 = thresh_df[thresh_df['ab_squared'] == ab_sq]
    if len(row95):
        r = row95.iloc[0]
        if not np.isnan(r['point_est']):
            ax.axvline(x=r['point_est'], color=COL_CUT, linestyle='-.',
                       linewidth=2.0, alpha=0.9,
                       label='Cut-off count' if show_leg else None)
            ax.text(r['point_est'] * 0.88, 0.0263,
                    f"{r['point_est']:.1f}",
                    color=COL_CUT, fontsize=9,
                    ha='right', fontweight='bold', va='bottom')

    ax.axvline(x=MIN_DISC_FOR_FIT, color='#444444', linestyle=':', linewidth=1.0, alpha=0.4)
    ax.set_xlabel('Expected Discordant Count per Class', fontsize=12)
    ax.set_ylabel('Type I Error Rate', fontsize=12)
    ax.set_title(f'$(a-b)^2 = {ab_sq}$', fontsize=13, fontweight='bold',
                 pad=18)
    ax.set_xscale('log')
    ax.set_ylim([0.025, 0.085])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.tick_params(axis='both', labelsize=10)
    ax.set_ylim([0.025, 0.085])
    ax.tick_params(axis='both', labelsize=10)

    if show_leg:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        auto_h, auto_l = ax.get_legend_handles_labels()
        handle_dict = dict(zip(auto_l, auto_h))

        h_strat    = handle_dict.get('Stratified')
        h_mcn      = handle_dict.get('McNemar')
        h_str_tr   = handle_dict.get('Stratified trend')
        h_mcn_tr   = handle_dict.get('McNemar trend')
        h_ci       = handle_dict.get('95% CI')
        h_ci_band  = handle_dict.get('95% CI band')
        h_cut      = handle_dict.get('Cut-off count')
        h_blank    = Patch(visible=False)

        ordered_h = [
            h_strat,   h_mcn,    h_blank,
            h_str_tr,   h_mcn_tr,  h_cut,
            h_ci,     h_ci_band,  h_blank
        ]
        ordered_l = [
            'PRI Method',   'McNemar',       '',
            'PRIM trend',
            'McNemar trend','Cut-off count',
            '95% CI trend',        '95% CI alpha',
            ''
        ]

        leg = ax.legend(ordered_h, ordered_l,
                        fontsize=10, loc='upper right',
                        frameon=True, framealpha=0.92, edgecolor='#cccccc',
                        markerscale=1.6, handlelength=2.5,
                        borderpad=0.9, labelspacing=0.6,
                        columnspacing=1.0,
                        ncol=3,
                        bbox_to_anchor=(1.0, 0.98),
                        bbox_transform=ax.transAxes)
        leg._legend_box.align = 'left'
    plt.tight_layout()
    fname = f'fig_6_expected_disc_absq{str(ab_sq).replace(".", "")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {fname}")


