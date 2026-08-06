"""
shap_publication_figures.py
===========================
SHAP analyses and confusion matrices in journal-ready quality, plus the numeric
SHAP values the manuscript tables are built from -- computed for BOTH cohorts.

Renamed from make_publication_figures.py: a second script of that name handles
the cohort/PCA/ROC figures, and one would silently overwrite the other in src/.

Place in:  src/            Run from:  src/  ->  python shap_publication_figures.py

WHAT CHANGED AGAINST THE PREVIOUS VERSION
-----------------------------------------
1. MULTICLASS FIX (affects the three-stage results in the manuscript).
   The old code passed `positive_class=1` for every task. For three_stage,
   TreeExplainer returns (n_samples, n_features, 3) and `sv[:, :, 1]` keeps
   only the MIDDLE class (F2/3) -- the smallest and least separable one. The
   reported attributions were therefore not "importance for the three-stage
   task" but "importance for predicting F2/3 specifically", which shrinks the
   magnitudes and reshuffles the ranking towards markers that happen to move
   that one class. Global multiclass importance is now the mean absolute SHAP
   value over samples AND classes, which is the standard definition.
   -> Re-check the three-stage row of the SHAP table against the new output.

2. BOTH COHORTS. SHAP is computed on the held-out UMM test partition and on the
   external MAINZ cohort. See the note below on what that does and does not
   mean. Cohort sizes are read from the data, never hard-coded.

3. CROSS-COHORT RANK AGREEMENT. Spearman correlation between the per-biomarker
   mean|SHAP| rankings of the two cohorts, per model and task, written to
   shap_rank_agreement.csv. One number per model/task answering: does the model
   rely on the same biomarkers in the external cohort?

4. NORMALISED SHARES. Absolute mean|SHAP| is not comparable across cohorts --
   the predicted-probability distributions differ because prevalence differs
   (50.0% vs 91.2% for moderate fibrosis). Each cohort's values are therefore
   additionally reported as a share of that cohort's total, so ranks and
   relative weights can be compared directly.

WHAT SHAP ON MAINZ MEANS
------------------------
The models are trained on UMM only. Running SHAP on MAINZ inputs explains the
SAME model on a different population: "which biomarkers drive this model's
predictions when it is applied externally?" That is the standard diagnostic for
whether a performance change under domain shift comes from the model leaning on
different features.

It does NOT say which biomarkers are important "in Mainz" as a biological
statement -- no model was trained there, so no such claim is available.

Both cohorts use TreeExplainer in its default tree_path_dependent mode, whose
reference distribution is the training data encoded in the trees. The baseline
is therefore identical for both cohorts and the comparison is well posed. Do
not switch one of them to the interventional mode with a cohort-specific
background; that would make the two sets incomparable.

Statistical note: the UMM values rest on ~31 patients, the MAINZ values on 284.
The +/- SD in the tables is the spread across the m=10 ensemble members, not
sampling uncertainty -- at this partition size the latter dominates. The external
are the more stable ones and are worth reporting as a robustness check even
where UMM stays the primary analysis.

Outputs (under outputs/figures/)
    shap_values_<model>_<task>_<cohort>.csv    mean|SHAP| per ensemble member
    shap_summary_<model>_<task>_<cohort>.pdf   beeswarm (vector)
    shap_bar_<model>_<task>_<cohort>.pdf       mean|SHAP| with SD across ensemble
    shap_rank_agreement.csv                    Spearman rho, UMM vs MAINZ
    shap_top5_table.tex                        LaTeX table, both cohorts
    confusion_<model>_<task>_<cohort>.pdf
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings('ignore')

FIGDIR = Path('outputs/figures')
SINGLE, DOUBLE = 89 / 25.4, 183 / 25.4

TASKS = ['fibrosis', 'two_stage', 'cirrhosis', 'three_stage']
TASK_LABEL = {'fibrosis': 'Moderate fibrosis', 'two_stage': 'Severe fibrosis',
              'cirrhosis': 'Cirrhosis', 'three_stage': 'Three-stage'}
TREE_MODELS = ['light_gbm', 'xgb', 'rf', 'svm']
MODEL_LABEL = {'light_gbm': 'LightGBM', 'xgb': 'XGBoost', 'rf': 'Random Forest',
               'svm': 'SVM'}
COHORTS = ['UMM', 'MAINZ']
TOP_N = 5

# Best ensemble per task and cohort -- the model whose confusion matrix goes into
# the combined 2x2 panel. Edit here if the ranking changes after a re-run.
# Selection rule: the model with the highest AUROC in the EXTERNAL cohort, applied
# to both cohorts. Rationale: with n=31 the internal ranking is not resolvable --
# the confidence intervals of all four models overlap almost completely, and the
# internal winner changes between tasks (SVM leads for cirrhosis). Picking per
# cohort would also mean the confusion panels show different models internally
# and externally, which makes them incomparable.
#
# Values from the stratified-split run (2026-08-02/03), external cohort:
#   fibrosis  LightGBM 0.865 | two_stage LightGBM 0.925 | cirrhosis VI-BNN 0.906
#
# For cirrhosis the nominal leader is VI-BNN, but the paired bootstrap on the
# same 284 patients gives dAUROC +0.018 (-0.012, +0.052), p=0.247 against Random
# Forest -- not separable. TabTransformer and LightGBM are likewise within noise;
# only XGBoost and below separate. Random Forest is therefore used, which keeps
# the model family consistent across all tasks. State that rule in the captions:
# "highest AUROC among models that are not separable from the leader by a paired
# test", not "highest AUROC".
#
# three_stage still needs recompute_three_stage.py to fix its entry.
_BEST_EXTERNAL = {'fibrosis': 'light_gbm', 'two_stage': 'light_gbm',
                  'cirrhosis': 'rf', 'three_stage': 'light_gbm'}   # <-- three_stage TBD
BEST_MODEL_PER_TASK = {'UMM': dict(_BEST_EXTERNAL), 'MAINZ': dict(_BEST_EXTERNAL)}
# The old one-file-per-model-and-task confusion matrices. Off by default now that
# the combined panels exist; set True if you need them for the supplement.
SAVE_INDIVIDUAL_CM = False
# Same for the one-file-per-model-and-task SHAP figures. The per-member CSVs are
# always written regardless of this flag.
SAVE_INDIVIDUAL_SHAP = False

SHAP_PANEL_TOP_N = 8
# Panels invite cross-panel comparison, but absolute mean|SHAP| is not comparable
# across tasks: the three-stage values are an order of magnitude smaller because
# the attribution is averaged over three classes. Normalising to each task's share
# of total attribution puts all four panels on one axis. Set False for absolute
# values matching the table, in which case every panel gets its own x-scale.
SHAP_PANEL_NORMALISE = True

PANEL_LETTERS = ['a', 'b', 'c', 'd']

# Combined ROC panel: one curve per task, using the same model selection as the
# confusion panels. Okabe-Ito palette (colour-blind safe, prints in greyscale).
# Muted, mid-dark palette. Saturated screen palettes (Okabe-Ito, matplotlib
# defaults) read as garish in print and lose separation in greyscale; these four
# differ in lightness as well as hue, so they stay distinguishable when the
# journal prints the figure in black and white.
ROC_COLOURS = {'fibrosis': '#2B4C6F',      # deep slate blue
               'two_stage': '#A2583D',     # muted terracotta
               'cirrhosis': '#4E7A5B',     # sage
               'three_stage': '#7A6E8F'}   # muted mauve
ROC_LINESTYLES = {'fibrosis': '-', 'two_stage': (0, (5.5, 1.8)),
                  'cirrhosis': (0, (1.4, 1.4)), 'three_stage': (0, (5, 1.6, 1.2, 1.6))}
# Bootstrap bands drawn behind the curves. Off by default: with four tasks on one
# axes and a small internal partition, the bands overlap into an unreadable wash.
# information is shown exactly once, in the AUROC strip below the curves.
ROC_CI_BANDS = False
# three_stage has no single ROC. Included as the macro-average of the three
# one-vs-rest curves and labelled as such; set False to leave it out.
ROC_INCLUDE_THREE_STAGE = True
N_BOOT, SEED = 1000, 0
N_UMM, N_MAINZ = 0, 0   # filled in main() from the data

CMAP = LinearSegmentedColormap.from_list('cb', ['#2166AC', '#B8B8B8', '#B2182B'])


def _tex(s):
    """Escape LaTeX specials in biomarker names. 'Quick (%)' and 'HbA1c (%)'
    would otherwise comment out the remainder of the table row."""
    s = str(s)
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                 ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')):
        s = s.replace(a, b)
    return s


def pub_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'figure.dpi': 300, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / f'{stem}.{ext}')
    plt.close(fig)
    print(f'    -> {FIGDIR}/{stem}.pdf (+.png)')


# ---------------------------------------------------------------- SHAP -----
def _as_array(sv):
    if isinstance(sv, list):                      # older shap: list of per-class arrays
        sv = np.stack(sv, axis=-1)
    return np.asarray(sv)


def shap_per_member(models, x_bg, x_explain, n_cls):
    """Per-ensemble-member SHAP values.

    Returns (per_member_signed, per_member_absmean):
        signed  -- (n_samples, n_features) for the plotted class; for multiclass
                   this is the top class (F4), used for the beeswarm only.
        absmean -- (n_features,) global importance: mean|SHAP| over samples, and
                   for multiclass additionally over classes.
    """
    import shap
    signed, absmean = [], []
    for mdl in models:
        try:
            sv = _as_array(shap.TreeExplainer(mdl).shap_values(x_explain))
        except Exception as exc:
            print(f'    TreeExplainer failed ({exc}); falling back to KernelExplainer')
            expl = shap.KernelExplainer(
                lambda d: mdl.predict_proba(d)[:, n_cls - 1], shap.sample(x_bg, 50))
            sv = _as_array(expl.shap_values(x_explain, silent=True))

        if sv.ndim == 3:                          # (n, f, C)
            absmean.append(np.abs(sv).mean(axis=(0, 2)))
            signed.append(sv[:, :, -1])           # top class: F4 / positive
        else:                                     # (n, f) -- binary, log-odds margin
            absmean.append(np.abs(sv).mean(axis=0))
            signed.append(sv)
    return signed, np.vstack(absmean)


def _beeswarm_ax(ax, sv, x, features, top_n=12, labelsize=6.5):
    order = np.argsort(np.abs(sv).mean(0))[::-1][:top_n][::-1]
    for row, fi in enumerate(order):
        vals, feat = sv[:, fi], x[:, fi]
        finite = np.isfinite(feat)
        lo, hi = (np.min(feat[finite]), np.max(feat[finite])) if finite.sum() > 1 else (0.0, 0.0)
        c = (feat - lo) / (hi - lo) if hi > lo else np.full_like(feat, 0.5, dtype=float)
        nb = 40
        edges = np.linspace(vals.min(), vals.max() + 1e-12, nb + 1)
        idx = np.clip(np.digitize(vals, edges) - 1, 0, nb - 1)
        y = np.zeros_like(vals, dtype=float)
        for b in range(nb):
            m = idx == b
            k = int(m.sum())
            if k:
                spread = min(0.36, 0.045 * np.sqrt(k))
                y[m] = np.linspace(-spread, spread, k) if k > 1 else 0.0
        ax.scatter(vals, row + y, c=c, cmap=CMAP, s=5, linewidths=0,
                   alpha=0.85, rasterized=True, vmin=0, vmax=1)
    ax.axvline(0, color='0.75', lw=0.5, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([features[i] for i in order], fontsize=labelsize)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    return order


def beeswarm(sv, x, features, title, stem, top_n=12):
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 0.19 * min(top_n, len(features)) + 0.9))
    _beeswarm_ax(ax, sv, x, features, top_n)
    ax.set_xlabel('SHAP value (impact on predicted probability)')
    ax.set_title(title, loc='left', fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.028, aspect=18)
    cb.set_ticks([0, 1]); cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Biomarker value', labelpad=-8)
    cb.outline.set_linewidth(0.4)
    save(fig, stem)


def bar_plot(mean_abs, sd_abs, features, title, stem, top_n=12):
    order = np.argsort(mean_abs)[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(SINGLE, 0.17 * len(order) + 0.8))
    ax.barh(range(len(order)), mean_abs[order], xerr=sd_abs[order],
            color='#4878A8', edgecolor='none', height=0.72,
            error_kw=dict(ecolor='0.35', lw=0.6, capsize=1.5))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([features[i] for i in order])
    ax.set_xlabel('mean |SHAP value|')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    save(fig, stem)


def shap_bar_panel(cohort, entries, stem, top_n=SHAP_PANEL_TOP_N,
                   normalise=SHAP_PANEL_NORMALISE):
    """Mean |SHAP| of the best model per task, four tasks in one 2x2 figure.

    entries: {task: (mean_abs, sd_abs, features, model_label)}
    """
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE * 0.94, DOUBLE * 0.66))
    xmax = 0
    for k, task in enumerate(TASKS):
        ax = axes[k // 2, k % 2]
        if task not in entries:
            ax.axis('off')
            continue
        mean_abs, sd_abs, features, model_label = entries[task]
        tot = mean_abs.sum()
        vals = mean_abs / tot * 100 if normalise else mean_abs
        errs = sd_abs / tot * 100 if normalise else sd_abs
        order = np.argsort(vals)[::-1][:top_n][::-1]
        ax.barh(range(len(order)), vals[order], xerr=errs[order],
                color=ROC_COLOURS[task], edgecolor='none', height=0.70,
                error_kw=dict(ecolor='0.45', lw=0.55, capsize=1.2))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([features[i] for i in order], fontsize=6.2)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_title(f'{PANEL_LETTERS[k]}  {TASK_LABEL[task]}', loc='left',
                     fontsize=7.5, fontweight='semibold', color='0.1', pad=11)
        ax.text(0, 1.028, model_label, transform=ax.transAxes, fontsize=6.1,
                color='0.45', va='bottom')
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('0.55')
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=6.2, colors='0.35', labelcolor='0.15')
        ax.grid(True, axis='x', lw=0.35, color='0.94')
        ax.set_axisbelow(True)
        xmax = max(xmax, float((vals[order] + errs[order]).max()))

    label = ('Share of total mean |SHAP| (%)' if normalise else 'mean |SHAP value|')
    for k, task in enumerate(TASKS):
        ax = axes[k // 2, k % 2]
        if task not in entries:
            continue
        if normalise:
            ax.set_xlim(0, xmax * 1.04)
        if k // 2 == 1:
            ax.set_xlabel(label, labelpad=2, fontsize=6.6, color='0.15')

    fig.suptitle(f'{cohort} cohort', x=0.004, y=0.995, ha='left',
                 fontsize=8.5, fontweight='semibold', color='0.1')
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, stem)


def shap_beeswarm_panel(cohort, entries, stem, top_n=SHAP_PANEL_TOP_N):
    """Beeswarm of the best model per task, four tasks in one 2x2 figure.

    entries: {task: (sv_signed, x, features, model_label)}
    """
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE * 0.98, DOUBLE * 0.70))
    for k, task in enumerate(TASKS):
        ax = axes[k // 2, k % 2]
        if task not in entries:
            ax.axis('off')
            continue
        sv, x, features, model_label = entries[task]
        _beeswarm_ax(ax, sv, x, features, top_n, labelsize=6.2)
        note = ' \u00b7 F4 class' if task == 'three_stage' else ''
        ax.set_title(f'{PANEL_LETTERS[k]}  {TASK_LABEL[task]}', loc='left',
                     fontsize=7.5, fontweight='semibold', color='0.1', pad=11)
        ax.text(0, 1.028, f'{model_label}{note}', transform=ax.transAxes,
                fontsize=6.1, color='0.45', va='bottom')
        ax.spines['bottom'].set_color('0.55')
        ax.tick_params(axis='x', labelsize=6.2, colors='0.35', labelcolor='0.15')
        if k // 2 == 1:
            ax.set_xlabel('SHAP value (impact on predicted probability)',
                          labelpad=2, fontsize=6.6, color='0.15')

    fig.suptitle(f'{cohort} cohort', x=0.004, y=0.995, ha='left',
                 fontsize=8.5, fontweight='semibold', color='0.1')
    fig.tight_layout(rect=(0, 0, 0.935, 0.975))
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cax = fig.add_axes([0.952, 0.13, 0.014, 0.68])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 1]); cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Biomarker value', labelpad=-10, fontsize=6.4)
    cb.ax.tick_params(labelsize=6.2)
    cb.outline.set_linewidth(0.4)
    save(fig, stem)


def bar_plot_paired(mean_umm, mean_mainz, features, title, stem, top_n=12):
    """Both cohorts side by side, on normalised shares so they are comparable."""
    su = mean_umm / mean_umm.sum() * 100
    sm = mean_mainz / mean_mainz.sum() * 100
    order = np.argsort(su)[::-1][:top_n][::-1]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(SINGLE * 1.25, 0.19 * len(order) + 0.9))
    ax.barh(y - 0.19, su[order], 0.36, color='#4878A8', edgecolor='none', label='UMM test')
    ax.barh(y + 0.19, sm[order], 0.36, color='#D65F5F', edgecolor='none', label='MAINZ')
    ax.set_yticks(y); ax.set_yticklabels([features[i] for i in order])
    ax.set_xlabel('Share of total mean |SHAP| (%)')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, loc='lower right')
    save(fig, stem)


# ------------------------------------------------------- confusion matrix ---
def confusion_plot(cm, classes, title, stem):
    cm = np.asarray(cm, dtype=float)
    pct = cm / np.clip(cm.sum(1, keepdims=True), 1, None) * 100
    fig, ax = plt.subplots(figsize=(SINGLE * 0.78, SINGLE * 0.78))
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{int(cm[i, j])}\n({pct[i, j]:.0f}%)', ha='center',
                    va='center', fontsize=6.5,
                    color='white' if pct[i, j] > 55 else '0.15')
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Observed')
    ax.set_title(title, loc='left', fontweight='bold')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, stem)


def _ensemble_proba(models, xs):
    """Soft-vote ensemble probability, member i evaluated on imputation i.

    That is the ensemble the manuscript describes. Evaluating every member on
    imputation 0 instead mixes a model with an imputation it never saw and
    understates the ensemble's agreement.
    """
    xs = xs if isinstance(xs, (list, tuple)) else [xs]
    probas = []
    for i, mdl in enumerate(models):
        x = np.asarray(xs[i] if i < len(xs) else xs[0])
        probas.append(np.asarray(mdl.predict_proba(x)))
    return np.mean(probas, axis=0)


def _roc_with_ci(y, score, n_boot=N_BOOT, seed=SEED):
    """ROC on a common FPR grid plus a percentile bootstrap band."""
    from sklearn.metrics import roc_curve, roc_auc_score
    y, score = np.asarray(y), np.asarray(score)
    grid = np.linspace(0, 1, 201)
    fpr, tpr, _ = roc_curve(y, score)
    base = np.interp(grid, fpr, tpr)
    base[0] = 0.0
    auc = roc_auc_score(y, score)

    rng = np.random.default_rng(seed)
    curves, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        f, t, _ = roc_curve(y[idx], score[idx])
        c = np.interp(grid, f, t); c[0] = 0.0
        curves.append(c)
        aucs.append(roc_auc_score(y[idx], score[idx]))
    if curves:
        band = (np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0))
        ci = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    else:
        band, ci = (base, base), (np.nan, np.nan)
    return grid, base, band, float(auc), ci


def _macro_ovr_score(proba, y, n_cls):
    """Macro-average one-vs-rest ROC input for the ordinal task.

    Returns (y_bin, score) stacked over the classes, which yields the
    macro-average curve when passed to roc_curve.
    """
    ys, ss = [], []
    for c in range(n_cls):
        ys.append((np.asarray(y) == c).astype(int))
        ss.append(proba[:, c])
    return np.concatenate(ys), np.concatenate(ss)


def roc_panel_combined(cohort, entries, stem):
    """Curves on top, AUROC with 95% bootstrap CI as a forest strip below.

    Layout follows the forest-plot convention: label column, whisker column,
    value column, each in its own axes, so nothing floats and the columns align
    down the figure. Confidence information is shown once, in the strip -- four
    overlapping bands in the ROC area cover most of the plot at this sample size
    and carry
    the same numbers less legibly.

    entries: {task: (y, score, n, model_label, is_macro)}
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    rows = [t for t in TASKS if t in entries]
    fig = plt.figure(figsize=(SINGLE * 1.62, SINGLE * 1.62 + 0.24 * len(rows) + 0.30))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.04 + 0.093 * len(rows)],
                  hspace=0.30, left=0.13, right=0.985, top=0.94, bottom=0.085)
    ax = fig.add_subplot(gs[0])
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1],
                                  width_ratios=[0.62, 1.0, 0.40], wspace=0.04)
    axl, axf, axv = (fig.add_subplot(sub[i]) for i in range(3))

    ax.plot([0, 1], [0, 1], ls=(0, (3, 3)), lw=0.6, c='0.72', zorder=1)
    stats_rows = []
    for task in rows:
        y, score, n, model_label, is_macro = entries[task]
        grid, base, band, auc, ci = _roc_with_ci(y, score)
        col = ROC_COLOURS[task]
        if ROC_CI_BANDS:
            ax.fill_between(grid, band[0], band[1], color=col, alpha=0.10,
                            linewidth=0, zorder=2)
        ax.plot(grid, base, lw=1.5, color=col, solid_capstyle='round',
                dash_capstyle='round', ls=ROC_LINESTYLES[task], zorder=3)
        stats_rows.append((task, col, auc, ci, model_label, is_macro))

    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, lw=0.35, color='0.94', zorder=0)
    ax.set_axisbelow(True)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('0.55')
    ax.tick_params(colors='0.35', labelcolor='0.15')
    ax.set_xlabel('1 \u2013 Specificity', labelpad=3, color='0.15')
    ax.set_ylabel('Sensitivity', labelpad=3, color='0.15')
    ax.set_title(f'{cohort} cohort  \u00b7  n = {entries[rows[0]][2]}', loc='left',
                 fontsize=8, fontweight='semibold', color='0.1', pad=6)

    handles = [plt.Line2D([], [], color=c, lw=1.5, ls=ROC_LINESTYLES[t])
               for t, c, _, _, _, _ in stats_rows]
    labels = [TASK_LABEL[t] + (' (macro OvR)' if mac else '')
              for t, _, _, _, _, mac in stats_rows]
    ax.legend(handles, labels, loc='lower right', frameon=False, fontsize=6.3,
              handlelength=2.2, labelspacing=0.42, borderaxespad=0.8,
              labelcolor='0.15')

    # ---- forest strip: label | whiskers | value ----
    ypos = np.arange(len(stats_rows))[::-1]
    lo_min = min(c[0] for _, _, _, c, _, _ in stats_rows)
    xlo = max(0.0, np.floor((lo_min - 0.03) * 20) / 20)

    for a in (axl, axv):
        a.set_xlim(0, 1); a.axis('off')
    for a in (axl, axf, axv):
        a.set_ylim(-0.6, len(stats_rows) - 0.4)

    for yp, (task, col, auc, ci, model_label, is_macro) in zip(ypos, stats_rows):
        axl.text(1.0, yp, f'{TASK_LABEL[task]}', ha='right', va='center',
                 fontsize=6.6, color='0.1')
        axl.text(1.0, yp - 0.30, model_label, ha='right', va='center',
                 fontsize=5.9, color='0.5')
        axf.plot([ci[0], ci[1]], [yp, yp], color=col, lw=1.1, alpha=0.85, zorder=2)
        for b in ci:
            axf.plot([b, b], [yp - 0.14, yp + 0.14], color=col, lw=1.0, zorder=3)
        axf.plot([auc], [yp], marker='o', ms=3.8, color=col,
                 markeredgecolor='white', markeredgewidth=0.7, zorder=4)
        axv.text(0.0, yp, f'{auc:.3f}', ha='left', va='center',
                 fontsize=6.6, color='0.1')
        axv.text(0.0, yp - 0.30, f'({ci[0]:.2f}\u2013{ci[1]:.2f})', ha='left',
                 va='center', fontsize=5.9, color='0.5')

    axf.set_yticks([])
    axf.set_xlim(xlo, 1.0)
    axf.set_xlabel('AUROC (95% bootstrap CI)', labelpad=3, fontsize=6.6, color='0.15')
    axf.grid(True, axis='x', lw=0.35, color='0.94')
    axf.set_axisbelow(True)
    axf.spines['left'].set_visible(False)
    axf.spines['bottom'].set_color('0.55')
    axf.tick_params(axis='y', length=0)
    axf.tick_params(axis='x', labelsize=6.2, colors='0.35', labelcolor='0.15')
    save(fig, stem)


def _cm_metrics(cm, y, pred):
    """Short performance line under each panel."""
    from sklearn.metrics import cohen_kappa_score
    cm = np.asarray(cm, dtype=float)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        sens = 100 * tp / (tp + fn) if (tp + fn) else np.nan
        spec = 100 * tn / (tn + fp) if (tn + fp) else np.nan
        return f'Sens. {sens:.1f}%   Spec. {spec:.1f}%'
    acc = 100 * np.trace(cm) / cm.sum() if cm.sum() else np.nan
    kap = cohen_kappa_score(y, pred, weights='linear')
    return f'Acc. {acc:.1f}%   $\\kappa_{{lin}}$ {kap:.3f}'


def confusion_panel(cohort, entries, stem):
    """All four tasks for one cohort in a single 2x2 figure.

    entries: {task: (cm, classes, n, model_label, metrics_line)}
    Cells are shaded by row-normalised percentage, i.e. by per-class recall, so
    the diagonal reads as sensitivity per observed class and the shading is not
    driven by class imbalance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE * 0.74, DOUBLE * 0.80))
    im = None
    for k, task in enumerate(TASKS):
        ax = axes[k // 2, k % 2]
        if task not in entries:
            ax.axis('off')
            continue
        cm, classes, n, model_label, metrics = entries[task]
        cm = np.asarray(cm, dtype=float)
        pct = cm / np.clip(cm.sum(1, keepdims=True), 1, None) * 100
        im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{int(cm[i, j])}\n{pct[i, j]:.0f}%', ha='center', va='center',
                        fontsize=7 if cm.shape[0] == 2 else 6.2,
                        color='white' if pct[i, j] > 55 else '0.15')
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, rotation=90, va='center')
        ax.set_xlabel('Predicted', labelpad=2)
        ax.set_ylabel('Observed', labelpad=2)
        ax.set_title(f'{PANEL_LETTERS[k]}  {TASK_LABEL[task]}', loc='left',
                     fontweight='bold', pad=14)
        ax.text(0, 1.045, f'{model_label} · n = {n} · {metrics}', transform=ax.transAxes,
                fontsize=6.3, color='0.35', va='bottom')
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle(f'{cohort} cohort', x=0.005, y=0.995, ha='left',
                 fontsize=9, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 0.93, 0.975))
    if im is not None:
        cax = fig.add_axes([0.945, 0.10, 0.017, 0.72])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Share of observed class (%)', labelpad=2)
        cb.outline.set_linewidth(0.4)
    save(fig, stem)


# ------------------------------------------------------------------ main ---
def main():
    pub_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    try:
        from src.preprocess import prepare_data
        from src.utils.ger_eng_dict import dict_germ_eng
    except ImportError:
        from preprocess import prepare_data
        from utils.ger_eng_dict import dict_germ_eng
    from sklearn.metrics import confusion_matrix

    table_rows, agreement_rows = [], []
    panels = {c: {} for c in COHORTS}
    roc_entries = {c: {} for c in COHORTS}
    shap_bars = {c: {} for c in COHORTS}
    shap_swarms = {c: {} for c in COHORTS}
    for task in TASKS:
        print(f'\n=== {task} ===')
        (xs_train, _, _, _, xs_test, ys_test,
         xs_pro, ys_pro, df_cols_de) = prepare_data(task, False, False)
        feats = [dict_germ_eng.get(c, c) for c in df_cols_de]
        n_cls = 3 if task == 'three_stage' else 2
        classes = ['F0-1', 'F2-3', 'F4'] if n_cls == 3 else ['Negative', 'Positive']
        data = {'UMM': (np.asarray(xs_test[0]), np.asarray(ys_test[0])),
                'MAINZ': (np.asarray(xs_pro[0]), np.asarray(ys_pro[0]))}
        imputations = {'UMM': xs_test, 'MAINZ': xs_pro}
        global N_UMM, N_MAINZ
        N_UMM, N_MAINZ = len(data['UMM'][1]), len(data['MAINZ'][1])

        for model in TREE_MODELS:
            path = Path(f'models/{model}/model_{task}.pickle')
            if not path.exists():
                print(f'  {model}: no checkpoint - skipped')
                continue
            with open(path, 'rb') as fh:
                models = pickle.load(fh)
            lbl = MODEL_LABEL.get(model, model)
            print(f'  {model}: {len(models)} ensemble members')

            means = {}
            for cohort in COHORTS:
                x, y = data[cohort]
                signed, abs_means = shap_per_member(
                    models, np.asarray(xs_train[0]), x, n_cls)
                mean_abs, sd_abs = abs_means.mean(0), abs_means.std(0, ddof=1)
                means[cohort] = mean_abs

                out = pd.DataFrame(abs_means, columns=feats)
                out.insert(0, 'ensemble_member', range(len(signed)))
                out.to_csv(FIGDIR / f'shap_values_{model}_{task}_{cohort.lower()}.csv',
                           index=False)

                # ensemble attribution: mean SHAP across members, not member 0
                sv_ens = np.mean(signed, axis=0)
                if SAVE_INDIVIDUAL_SHAP:
                    cls_note = ' (F4 class)' if n_cls == 3 else ''
                    beeswarm(sv_ens, x, feats,
                             f'{TASK_LABEL[task]} - {lbl} - {cohort}{cls_note}',
                             f'shap_summary_{model}_{task}_{cohort.lower()}')
                    bar_plot(mean_abs, sd_abs, feats,
                             f'{TASK_LABEL[task]} - {lbl} - {cohort}',
                             f'shap_bar_{model}_{task}_{cohort.lower()}')
                if BEST_MODEL_PER_TASK.get(cohort, {}).get(task) == model:
                    shap_bars[cohort][task] = (mean_abs, sd_abs, feats, lbl)
                    shap_swarms[cohort][task] = (sv_ens, x, feats, lbl)

                ranks = np.empty(len(feats), dtype=int)
                ranks[np.argsort(mean_abs)[::-1]] = np.arange(1, len(feats) + 1)
                for i in range(len(feats)):
                    table_rows.append(dict(task=task, model=lbl, cohort=cohort,
                                           feature=feats[i], rank=int(ranks[i]),
                                           mean_abs_shap=mean_abs[i], sd=sd_abs[i]))

                proba = _ensemble_proba(models, imputations[cohort])
                pred = proba.argmax(1)
                cm = confusion_matrix(y, pred, labels=range(n_cls))
                if SAVE_INDIVIDUAL_CM:
                    confusion_plot(cm, classes, f'{TASK_LABEL[task]} - {lbl} ({cohort})',
                                   f'confusion_{model}_{task}_{cohort.lower()}')
                if BEST_MODEL_PER_TASK.get(cohort, {}).get(task) == model:
                    panels[cohort][task] = (cm, classes, len(y), lbl,
                                            _cm_metrics(cm, y, pred))
                    if n_cls == 2:
                        roc_entries[cohort][task] = (y, proba[:, 1], len(y), lbl, False)
                    elif ROC_INCLUDE_THREE_STAGE:
                        yb, sc = _macro_ovr_score(proba, y, n_cls)
                        roc_entries[cohort][task] = (yb, sc, len(y), lbl, True)

            if len(means) == 2:
                rho = stats.spearmanr(means['UMM'], means['MAINZ']).statistic
                top_u = set(np.argsort(means['UMM'])[::-1][:TOP_N])
                top_m = set(np.argsort(means['MAINZ'])[::-1][:TOP_N])
                agreement_rows.append(dict(
                    task=task, model=lbl, spearman_rho=round(float(rho), 3),
                    top5_overlap=f'{len(top_u & top_m)}/{TOP_N}',
                    shared_top5=', '.join(sorted(feats[i] for i in top_u & top_m))))
                print(f'    UMM vs MAINZ: rho={rho:.3f}, '
                      f'top-5 overlap {len(top_u & top_m)}/{TOP_N}')
                bar_plot_paired(means['UMM'], means['MAINZ'], feats,
                                f'{TASK_LABEL[task]} - {lbl}',
                                f'shap_paired_{model}_{task}')

    for cohort in COHORTS:
        if panels[cohort]:
            missing = [t for t in TASKS if t not in panels[cohort]]
            if missing:
                print(f'\n{cohort} panel: no checkpoint for '
                      f'{", ".join(BEST_MODEL_PER_TASK[cohort][t] + "/" + t for t in missing)} '
                      f'-- those cells stay empty')
            confusion_panel(cohort, panels[cohort],
                            f'confusion_panel_{cohort.lower()}')
        if roc_entries[cohort]:
            roc_panel_combined(cohort, roc_entries[cohort],
                               f'roc_panel_{cohort.lower()}')
        if shap_bars[cohort]:
            shap_bar_panel(cohort, shap_bars[cohort],
                           f'shap_bar_panel_{cohort.lower()}')
            shap_beeswarm_panel(cohort, shap_swarms[cohort],
                                f'shap_beeswarm_panel_{cohort.lower()}')

    if agreement_rows:
        ag = pd.DataFrame(agreement_rows)
        ag.to_csv(FIGDIR / 'shap_rank_agreement.csv', index=False)
        print('\nCross-cohort rank agreement:')
        print(ag.to_string(index=False))

    if table_rows:
        df = pd.DataFrame(table_rows)
        df.to_csv(FIGDIR / 'shap_all_features.csv', index=False)
        df[df['rank'] <= TOP_N].to_csv(FIGDIR / 'shap_top5.csv', index=False)
        ag = {(r['task'], r['model']): r for r in agreement_rows}

        lines = [r'\begin{table*}[htbp]', r'    \centering',
                 r'    \caption{\small{Five most influential biomarkers per task for the',
                 r'    best-performing model. The two cohorts are ranked independently: the left',
                 f'    block gives the top five on the held-out UMM test partition (n={N_UMM}), the',
                 f'    right block the top five on the external MAINZ cohort (n={N_MAINZ}), each by the',
                 r'    mean absolute SHAP value of the ensemble. Values are mean $\pm$ standard',
                 r'    deviation across the $m=10$ ensemble members; the number in parentheses is',
                 r'    that biomarker\'s rank in the respective other cohort. $\rho$ is the Spearman',
                 r'    correlation between the two full rankings over all biomarkers.}}',
                 r'    \label{tab:shap_top5}',
                 r'    \begin{tabular}{clclc}', r'        \toprule',
                 r'        & \multicolumn{2}{c}{\textbf{UMM test partition}}'
                 r' & \multicolumn{2}{c}{\textbf{MAINZ cohort}}\\',
                 r'        \cmidrule(lr){2-3}\cmidrule(lr){4-5}',
                 r'        \textbf{Rank} & \textbf{Biomarker} & \textbf{mean $|$SHAP$|$}'
                 r' & \textbf{Biomarker} & \textbf{mean $|$SHAP$|$}\\',
                 r'        \midrule']

        for task in TASKS:
            sub = df[df.task == task]
            if sub.empty:
                continue
            # Use the model named in BEST_MODEL_PER_TASK, not whichever row happens
            # to come first. sub.model.iloc[0] was always 'LightGBM' because
            # table_rows is filled in TREE_MODELS order -- which made Table 4 and
            # Figures 7-10 report different models for the same task.
            best_key = BEST_MODEL_PER_TASK['MAINZ'].get(task)
            best = MODEL_LABEL.get(best_key, best_key)
            if best not in set(sub.model):
                best = sub.model.iloc[0]
                print(f'  Table 4/{task}: {best_key} not among TREE_MODELS, '
                      f'falling back to {best}')
            sub = sub[sub.model == best]
            u = sub[sub.cohort == 'UMM'].set_index('feature')
            m = sub[sub.cohort == 'MAINZ'].set_index('feature')
            u_top = u.sort_values('rank').head(TOP_N)
            m_top = m.sort_values('rank').head(TOP_N)

            head = f'\\textit{{{TASK_LABEL[task]}}} ({best})'
            info = ag.get((task, best))
            if info:
                head += (f' --- $\\rho$ = {info["spearman_rho"]:.3f}, '
                         f'top-5 overlap {info["top5_overlap"]}')
            lines.append(f'        \\multicolumn{{5}}{{l}}{{{head}}}\\\\')

            for k in range(TOP_N):
                fu, fm = u_top.index[k], m_top.index[k]
                ru = f'{_tex(fu)} ({int(m.loc[fu, "rank"])})' if fu in m.index else _tex(fu)
                rm = f'{_tex(fm)} ({int(u.loc[fm, "rank"])})' if fm in u.index else _tex(fm)
                lines.append(
                    f'        {k + 1} & {ru} & '
                    f'{u_top.iloc[k].mean_abs_shap:.3f} $\\pm$ {u_top.iloc[k].sd:.3f} & '
                    f'{rm} & '
                    f'{m_top.iloc[k].mean_abs_shap:.3f} $\\pm$ {m_top.iloc[k].sd:.3f}\\\\')
            lines.append(r'        \midrule')

        lines[-1] = r'        \bottomrule'
        lines += [r'    \end{tabular}', r'\end{table*}']
        (FIGDIR / 'shap_top5_table.tex').write_text('\n'.join(lines), encoding='utf-8')
        print(f'\nLaTeX table -> {FIGDIR}/shap_top5_table.tex')
        print(f'Full per-feature values -> {FIGDIR}/shap_all_features.csv')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()