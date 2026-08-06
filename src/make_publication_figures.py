"""
make_publication_figures.py
===========================
Journal-ready figures and tables for the revised manuscript.

    1. Fibrosis-stage histogram, UMM vs. MAINZ
    2. PCA domain-shift plot, UMM vs. MAINZ (no closest-patient overlay)
    3. ROC curves per task, AUROC + 95% bootstrap CI in the legend
    4. Patient-characteristics table

SINGLE SOURCE OF TRUTH
----------------------
Every cohort number in this script comes from `prepare_data()` -- the same
entry point `train.py` uses. Nothing is read from a pre-computed CSV snapshot
and no cohort size is hard-coded, so changing `window_days_pre` (or any other
preprocessing switch) propagates here automatically. If the analytic cohort
moves from 304 to some other n, this script follows without an edit.

Place in:  src/          Run from:  src/   ->   python make_publication_figures.py
                         or from repo root ->   python src/make_publication_figures.py

WHAT prepare_data() CAN AND CANNOT PROVIDE
------------------------------------------
It returns the *imputed* (optionally scaled) design matrices plus the task
label. That covers the PCA, the ROC curves, the cohort sizes and the imputed
columns of the characteristics table.

It does NOT carry the pre-imputation values, so "measured mean +/- SD" and
"Missing (%)" cannot be derived from it. Those two columns require a raw
snapshot written *during the same preprocessing run*. Set RAW_SNAPSHOT below
if you have one (see the note at the bottom of this file for the six lines
that write it); otherwise the table is emitted with the imputed columns only
and the caption is adjusted accordingly -- rather than silently reporting 0.0%
missingness, which is what the previous version did whenever it fell back.

Fibrosis grades: `prepare_data` returns task labels, not F0-F4. The grade
histogram is therefore reconstructed by combining the labels of the four tasks
(see `derive_grades`), which recovers F0/1, F2, F3 and F4. F0 and F1 cannot be
separated this way; if you need them apart, supply RAW_SNAPSHOT.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

# --------------------------------------------------------------- config -----
SRC_DIR = Path(__file__).resolve().parent
FIGDIR = SRC_DIR / 'outputs' / 'figures'
MODELS_DIR = SRC_DIR / 'models'
DATA_DIR = SRC_DIR.parent / 'data'
ATTRITION_JSON = SRC_DIR / 'outputs' / 'data_qc' / 'patient_attrition.json'

# Optional: raw (pre-imputation) snapshot written by the *current* run.
# Must contain a 'cohort' column with values 'umm' / 'mainz' and, ideally, a
# fibrosis-grade column named 'Micro_raw' or 'F_stage'. None -> skip raw cols.
RAW_SNAPSHOT = None          # e.g. SRC_DIR / 'outputs' / 'data_qc' / 'raw_cohort_snapshot.csv'

# Task used to define the cohort and the row order for PCA / characteristics.
PRIMARY_TASK = 'three_stage'
ALL_TASKS = ['fibrosis', 'two_stage', 'cirrhosis', 'three_stage']

ROC_TASKS = {'fibrosis': 'Moderate Fibrosis',
             'two_stage': 'Severe Fibrosis',
             'cirrhosis': 'Cirrhosis'}
ROC_MODELS = {'light_gbm': 'LightGBM', 'xgb': 'XGBoost', 'rf': 'Random Forest',
              'svm': 'SVM', 'vi_bnn': 'VI-BNN'}

N_BOOT, SEED = 1000, 0
SINGLE, DOUBLE = 89 / 25.4, 183 / 25.4
UMM_COL, MAINZ_COL = '#4878A8', '#D65F5F'


def _prepare_data():
    """Import prepare_data whether the script runs from src/ or from the repo root."""
    if str(SRC_DIR.parent) not in sys.path:
        sys.path.insert(0, str(SRC_DIR.parent))
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from src.preprocess import prepare_data          # same import train.py uses
    except ImportError:
        from preprocess import prepare_data
    return prepare_data


def pub_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
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
    print(f'  -> {FIGDIR}/{stem}.pdf (+.png)')


# ============================================================== loading =====
_TASK_CACHE = {}


def load_task(task, scaling=False):
    """Return the full cohort for one task, straight out of prepare_data().

    UMM rows are train + val + test in that fixed order, so the row order is
    identical across tasks as long as the split is deterministic (verified in
    `derive_grades`).
    """
    key = (task, scaling)
    if key in _TASK_CACHE:
        return _TASK_CACHE[key]
    prepare_data = _prepare_data()
    d = prepare_data(task, False, scaling)
    xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, cols = d

    m = len(xs_train)
    X_umm = np.stack([np.vstack([np.asarray(xs_train[i]),
                                 np.asarray(xs_val[i]),
                                 np.asarray(xs_test[i])]) for i in range(m)])
    y_umm = np.concatenate([np.asarray(ys_train[0]).ravel(),
                            np.asarray(ys_val[0]).ravel(),
                            np.asarray(ys_test[0]).ravel()])
    X_mainz = np.stack([np.asarray(xs_pro[i]) for i in range(m)])
    y_mainz = np.asarray(ys_pro[0]).ravel()

    out = {
        'task': task, 'm': m, 'cols': list(cols),
        'X_umm': X_umm, 'y_umm': y_umm,
        'X_mainz': X_mainz, 'y_mainz': y_mainz,
        'n_train': len(np.asarray(ys_train[0]).ravel()),
        'n_val': len(np.asarray(ys_val[0]).ravel()),
        'n_test': len(np.asarray(ys_test[0]).ravel()),
        'raw': d,
    }
    _TASK_CACHE[key] = out
    return out


def check_split_overlap(task):
    """Leakage check: prepare_data splits the RAW frame before the pre-biopsy
    filter runs, and the per-patient de-duplication happens inside each split.
    A patient with two eligible laboratory rows can therefore end up in two
    partitions. Compare the pre-imputation split exports row-wise."""
    frames = {}
    for split in ('train', 'val', 'test'):
        p = DATA_DIR / f'preprocessed_no_mice_{split}' / f'{split}_{task}.csv'
        if not p.exists():
            print(f'  split-overlap check: {p} not found -- skipped')
            return None
        frames[split] = pd.read_csv(p)

    def keys(df):
        return set(map(tuple, df.round(6).fillna(-999999).to_numpy().tolist()))

    k = {s: keys(df) for s, df in frames.items()}
    total = sum(len(df) for df in frames.values())
    overlaps = {f'{a}/{b}': len(k[a] & k[b])
                for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    dupes = total - len(k['train'] | k['val'] | k['test'])
    print(f"  UMM rows across splits: {total} "
          f"(train {len(frames['train'])} / val {len(frames['val'])} / test {len(frames['test'])})")
    if any(overlaps.values()):
        print(f'  !! identical rows shared between partitions: {overlaps} '
              f'-- patient-level leakage, de-duplicate before splitting')
    else:
        print(f'  no identical rows shared between partitions ({dupes} duplicates within splits)')
    return total


def report_cohort(c):
    n_umm, n_mainz = c['X_umm'].shape[1], c['X_mainz'].shape[1]
    print(f"  cohort from prepare_data('{c['task']}'): "
          f"UMM n={n_umm} (train {c['n_train']} / val {c['n_val']} / test {c['n_test']}), "
          f"MAINZ n={n_mainz}, imputations m={c['m']}, biomarkers={len(c['cols'])}")

    if ATTRITION_JSON.exists():
        import json
        att = json.loads(ATTRITION_JSON.read_text())
        final = att.get('final analytic cohort')
        if final is not None and int(final) != n_umm:
            print(f'  !! patient_attrition.json says {final}, prepare_data yields {n_umm}. '
                  f'The attrition cascade de-duplicates globally, the pipeline per split -- '
                  f'this is the 304-vs-305 discrepancy. Pick one and fix the cause.')
        elif final is not None:
            print(f'  attrition cascade agrees: final analytic cohort = {final}')

    check_split_overlap(c['task'])

    (FIGDIR / 'cohort_sizes.txt').write_text(
        f"task={c['task']}\numm_total={n_umm}\numm_train={c['n_train']}\n"
        f"umm_val={c['n_val']}\numm_test={c['n_test']}\nmainz={n_mainz}\n"
        f"imputations={c['m']}\nbiomarkers={len(c['cols'])}\n", encoding='utf-8')
    return n_umm, n_mainz


# ============================================================== grades ======
def derive_grades(scaling=False):
    """Reconstruct fibrosis grades from the four task labels.

        three_stage: 0 = F0/1, 1 = F2/3, 2 = F4
        two_stage:   1 = F3/4        -> separates F3 from F2 inside class 1
        cirrhosis:   1 = F4          -> cross-check for three_stage class 2

    Recovers F0/1 (merged), F2, F3, F4. Returns (umm_labels, mainz_labels) as
    arrays of strings, or (None, None) if the task splits are not aligned.
    """
    cohorts = {}
    for t in ALL_TASKS:
        try:
            cohorts[t] = load_task(t, scaling)
        except Exception as exc:
            print(f'  grade derivation: prepare_data("{t}") failed ({exc})')
            return None, None

    ref = cohorts[PRIMARY_TASK]
    for t, c in cohorts.items():
        if c['X_umm'].shape[1] != ref['X_umm'].shape[1]:
            print(f'  grade derivation: cohort size differs between '
                  f'"{t}" ({c["X_umm"].shape[1]}) and "{PRIMARY_TASK}" '
                  f'({ref["X_umm"].shape[1]}) -- skipped')
            return None, None
        if not np.allclose(c['X_umm'][0], ref['X_umm'][0], equal_nan=True):
            print(f'  grade derivation: row order differs between "{t}" and '
                  f'"{PRIMARY_TASK}" -- skipped (splits are not aligned)')
            return None, None

    out = []
    for key_y in ('y_umm', 'y_mainz'):
        three = cohorts['three_stage'][key_y].astype(int)
        two = cohorts['two_stage'][key_y].astype(int)
        cirr = cohorts['cirrhosis'][key_y].astype(int)
        lab = np.empty(len(three), dtype=object)
        lab[three == 0] = 'F0/1'
        lab[(three == 1) & (two == 0)] = 'F2'
        lab[(three == 1) & (two == 1)] = 'F3'
        lab[three == 2] = 'F4'
        mismatch = int(np.sum((three == 2) != (cirr == 1)))
        if mismatch:
            print(f'  grade derivation ({key_y}): {mismatch} label mismatches '
                  f'between three_stage and cirrhosis -- check the task definitions')
        out.append(lab)
    return out[0], out[1]


def grades_from_snapshot(raw):
    for col in ('F_stage', 'Micro_raw', 'Fibrosis'):
        if col in raw.columns:
            g = pd.to_numeric(raw[col], errors='coerce')
            return g.map(lambda v: f'F{int(v)}' if pd.notna(v) else None)
    return None


# =========================================================== stage hist =====
def stage_histograms(umm_labels, mainz_labels, order):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
    for ax, data, title, col in [(axes[0], umm_labels, 'UMM', UMM_COL),
                                 (axes[1], mainz_labels, 'MAINZ', MAINZ_COL)]:
        s = pd.Series(list(data)).dropna()
        counts = s.value_counts().reindex(order, fill_value=0)
        ax.bar(range(len(order)), counts.values, color=col,
               edgecolor='white', linewidth=0.6, width=0.8)
        for i, v in enumerate(counts.values):
            if v:
                ax.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=6)
        ax.set_title(f'{title} (n={int(counts.sum())})', loc='left', fontweight='bold')
        ax.set_xlabel('Fibrosis stage')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_ylabel('Number of patients')
    fig.tight_layout()
    save(fig, 'stage_histograms_analytic_cohort')

    tab = pd.DataFrame({
        'stage': order,
        'UMM': [int(pd.Series(list(umm_labels)).eq(g).sum()) for g in order],
        'MAINZ': [int(pd.Series(list(mainz_labels)).eq(g).sum()) for g in order]})
    tab.to_csv(FIGDIR / 'stage_distribution.csv', index=False)
    print('  stage counts (paste into Patient Characteristics):')
    print(tab.to_string(index=False))


# ================================================================= PCA ======
def _ellipse(ax, x, y, col, nstd=2.0):
    if len(x) < 3:
        return
    vals, vecs = np.linalg.eigh(np.cov(x, y))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), w, h, angle=theta,
                         facecolor=col, alpha=0.12, edgecolor=col, lw=1.0, ls='--'))


STAGE_ORDER = ['F0/1', 'F2', 'F3', 'F4']
STAGE_COLS = {'F0/1': '#FEE08B', 'F2': '#FDAE61', 'F3': '#E4633A', 'F4': '#A50026'}


def _pca_fit(X_umm, X_mainz):
    both = np.vstack([X_umm, X_mainz])
    scaler = StandardScaler().fit(both)
    pca = PCA(n_components=2).fit(scaler.transform(both))
    return (pca.transform(scaler.transform(X_umm)),
            pca.transform(scaler.transform(X_mainz)),
            pca.explained_variance_ratio_ * 100)


def pca_stage_separation(pu, pm, umm_lab, mainz_lab):
    """Quantify what the eye is asked to see: does stage structure PC space,
    and does it do so more or less than cohort membership?"""
    from sklearn.metrics import silhouette_score
    P = np.vstack([pu, pm])
    stage = np.concatenate([np.asarray(umm_lab), np.asarray(mainz_lab)])
    cohort = np.array(['UMM'] * len(pu) + ['MAINZ'] * len(pm))
    ordinal = np.array([STAGE_ORDER.index(s) if s in STAGE_ORDER else np.nan for s in stage],
                       dtype=float)
    ok = ~np.isnan(ordinal)

    rows = [{'grouping': 'cohort (UMM vs MAINZ)', 'scope': 'pooled',
             'silhouette_PC1_PC2': round(float(silhouette_score(P, cohort)), 3),
             'spearman_PC1': '--', 'kruskal_p_PC1': '--'}]
    for name, mask in (('pooled', ok),
                       ('UMM', ok & (cohort == 'UMM')),
                       ('MAINZ', ok & (cohort == 'MAINZ'))):
        lab, pc1 = stage[mask], P[mask, 0]
        if len(np.unique(lab)) < 2:
            continue
        rho = stats.spearmanr(pc1, ordinal[mask]).statistic
        groups = [pc1[lab == s] for s in STAGE_ORDER if (lab == s).sum() > 1]
        kw = stats.kruskal(*groups).pvalue if len(groups) > 1 else np.nan
        rows.append({'grouping': 'fibrosis stage (F0/1-F4)', 'scope': name,
                     'silhouette_PC1_PC2': round(float(silhouette_score(P[mask], lab)), 3),
                     'spearman_PC1': round(float(rho), 3),
                     'kruskal_p_PC1': ('<0.001' if kw < 0.001 else round(float(kw), 4))})
        cirr = np.where(lab == 'F4', 'F4', 'F0-F3')
        rows.append({'grouping': 'cirrhosis (F4 vs F0-F3)', 'scope': name,
                     'silhouette_PC1_PC2': round(float(silhouette_score(P[mask], cirr)), 3),
                     'spearman_PC1': '--',
                     'kruskal_p_PC1': ('<0.001' if stats.mannwhitneyu(
                         P[mask][cirr == 'F4', 0], P[mask][cirr == 'F0-F3', 0]).pvalue < 0.001
                         else round(float(stats.mannwhitneyu(
                             P[mask][cirr == 'F4', 0], P[mask][cirr == 'F0-F3', 0]).pvalue), 4))})
    table = pd.DataFrame(rows)
    table.to_csv(FIGDIR / 'pca_stage_separation.csv', index=False)
    print('  separation in PC1/PC2 space:')
    print(table.to_string(index=False))
    return table


def pca_domain_shift(X_umm, X_mainz, umm_lab=None, mainz_lab=None):
    """Three panels on one shared projection:
       (a) cohort membership -- the domain shift
       (b) fibrosis stage    -- does severity structure the same space?
       (c) cirrhosis yes/no  -- the single decision that separates best
    Falls back to panel (a) alone if no stage labels are supplied."""
    pu, pm, ev = _pca_fit(X_umm, X_mainz)
    have = umm_lab is not None and mainz_lab is not None
    ncol = 3 if have else 1
    fig, axes = plt.subplots(1, ncol, figsize=(DOUBLE if have else SINGLE * 1.35,
                                               SINGLE * 1.32), squeeze=False)
    axes = axes[0]

    def frame(ax, title, ncol_legend=1):
        ax.set_xlabel(f'PC1 ({ev[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({ev[1]:.1f}%)')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.legend(loc='best', frameon=False, handletextpad=0.3, borderaxespad=0.2,
                  ncol=ncol_legend, columnspacing=0.8, fontsize=6, markerscale=1.4)

    # (a) cohort
    ax = axes[0]
    ax.scatter(pu[:, 0], pu[:, 1], s=8, c=UMM_COL, alpha=0.7, linewidths=0,
               label=f'UMM (n={len(pu)})', rasterized=True)
    ax.scatter(pm[:, 0], pm[:, 1], s=8, c=MAINZ_COL, alpha=0.7, linewidths=0,
               marker='^', label=f'MAINZ (n={len(pm)})', rasterized=True)
    _ellipse(ax, pu[:, 0], pu[:, 1], UMM_COL)
    _ellipse(ax, pm[:, 0], pm[:, 1], MAINZ_COL)
    frame(ax, 'a  Cohort')

    if not have:
        fig.tight_layout()
        save(fig, 'pca_domain_shift_umm_mainz')
        return

    P = np.vstack([pu, pm])
    stage = np.concatenate([np.asarray(umm_lab), np.asarray(mainz_lab)])
    marker = np.array(['o'] * len(pu) + ['^'] * len(pm))

    # (b) stage, with per-stage centroids
    ax = axes[1]
    for s in STAGE_ORDER:
        for mk in ('o', '^'):
            m = (stage == s) & (marker == mk)
            if m.sum():
                ax.scatter(P[m, 0], P[m, 1], s=8, c=STAGE_COLS[s], marker=mk,
                           alpha=0.75, linewidths=0, rasterized=True,
                           label=f'{s} (n={int((stage == s).sum())})' if mk == 'o' else None)
    for s in STAGE_ORDER:
        m = stage == s
        if m.sum():
            ax.scatter(P[m, 0].mean(), P[m, 1].mean(), s=42, c=STAGE_COLS[s],
                       marker='D', edgecolors='black', linewidths=0.7, zorder=5)
    frame(ax, 'b  Fibrosis stage', ncol_legend=2)

    # (c) cirrhosis
    ax = axes[2]
    cirr = stage == 'F4'
    ax.scatter(P[~cirr, 0], P[~cirr, 1], s=8, c='#9FB6C6', alpha=0.7, linewidths=0,
               label=f'F0-F3 (n={int((~cirr).sum())})', rasterized=True)
    ax.scatter(P[cirr, 0], P[cirr, 1], s=8, c='#A50026', alpha=0.75, linewidths=0,
               label=f'F4 (n={int(cirr.sum())})', rasterized=True)
    _ellipse(ax, P[~cirr, 0], P[~cirr, 1], '#9FB6C6')
    _ellipse(ax, P[cirr, 0], P[cirr, 1], '#A50026')
    frame(ax, 'c  Cirrhosis')

    fig.tight_layout()
    save(fig, 'pca_cohort_stage_cirrhosis')
    pca_stage_separation(pu, pm, umm_lab, mainz_lab)


# ================================================= patient characteristics ==
def _tex(s):
    """Escape LaTeX specials in biomarker names -- 'Quick (%)' and 'HbA1c (%)'
    would otherwise comment out the rest of the row."""
    s = str(s)
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                 ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'),
                 ('}', r'\}'), ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')):
        s = s.replace(a, b)
    return s


def characteristics_table(cols, X_umm, X_mainz, raw=None):
    """X_* are (m, n, p) imputed stacks. `raw` is the optional snapshot frame."""
    umm_mean = X_umm.mean(axis=0)      # per-patient mean across imputations
    mainz_mean = X_mainz.mean(axis=0)
    have_raw = raw is not None and 'cohort' in raw.columns
    if have_raw:
        raw_umm = raw[raw.cohort == 'umm']
        raw_mainz = raw[raw.cohort == 'mainz']

    rows = []
    for j, feat in enumerate(cols):
        u_imp, m_imp = umm_mean[:, j], mainz_mean[:, j]
        row = {'Characteristic': feat,
               'UMM (imputed)': f'{np.nanmean(u_imp):.1f} $\\pm$ {np.nanstd(u_imp, ddof=1):.1f}',
               'MAINZ (imputed)': f'{np.nanmean(m_imp):.1f} $\\pm$ {np.nanstd(m_imp, ddof=1):.1f}'}
        if have_raw and feat in raw_umm.columns:
            u = pd.to_numeric(raw_umm[feat], errors='coerce')
            mm = pd.to_numeric(raw_mainz[feat], errors='coerce') if feat in raw_mainz.columns else pd.Series(dtype=float)
            ud, md = u.dropna(), mm.dropna()
            row['UMM (measured)'] = f'{ud.mean():.1f} $\\pm$ {ud.std():.1f}' if len(ud) else '--'
            row['MAINZ (measured)'] = f'{md.mean():.1f} $\\pm$ {md.std():.1f}' if len(md) else '--'
            row['Missing (\\%)'] = f'{u.isna().mean() * 100:.1f}'
            src_u, src_m = ud, md
        else:
            src_u, src_m = pd.Series(u_imp).dropna(), pd.Series(m_imp).dropna()
        if len(src_u) > 1 and len(src_m) > 1:
            p = stats.ttest_ind(src_u, src_m, equal_var=False).pvalue
            row['p-value'] = '<0.001' if p < 0.001 else f'{p:.3f}'
        else:
            row['p-value'] = '--'
        rows.append(row)

    df = pd.DataFrame(rows)
    if have_raw:
        order = ['Characteristic', 'UMM (measured)', 'MAINZ (measured)',
                 'UMM (imputed)', 'MAINZ (imputed)', 'p-value', 'Missing (\\%)']
        note = ('Values are mean $\\pm$ standard deviation. Measured columns use non-imputed '
                'values; imputed columns are averaged over the $m$ multiply imputed datasets. '
                'P-values are Welch two-sample t-tests on the measured values. '
                'Missing (\\%) refers to the UMM cohort before imputation.')
    else:
        order = ['Characteristic', 'UMM (imputed)', 'MAINZ (imputed)', 'p-value']
        note = ('Values are mean $\\pm$ standard deviation after multiple imputation, averaged '
                'over the $m$ imputed datasets. P-values are Welch two-sample t-tests on the '
                'imputed values and are therefore anti-conservative; pre-imputation values and '
                'missingness rates were not available to this script.')
        print('  NOTE: no raw snapshot -> measured and missingness columns omitted '
              '(see the footer of this file for how to write one).')

    df = df[order]
    df.to_csv(FIGDIR / 'patient_characteristics.csv', index=False)

    n_u, n_m = X_umm.shape[1], X_mainz.shape[1]
    spec = 'l' + 'c' * (len(order) - 1)
    lines = [r'\begin{table*}[htbp]', r'    \centering',
             f'    \\caption{{\\small{{Patient characteristics of the UMM (n={n_u}) and '
             f'MAINZ (n={n_m}) cohorts. {note}}}}}',
             r'    \label{tab:patient_characteristics}',
             f'    \\begin{{tabular}}{{{spec}}}', r'        \toprule',
             '        ' + ' & '.join(f'\\textbf{{{c}}}' for c in order) + r'\\',
             r'        \midrule']
    for i, (_, r) in enumerate(df.iterrows()):
        sh = r'\rowcolor{gray!10} ' if i % 2 == 0 else ''
        cells = [_tex(r[c]) if c == 'Characteristic' else str(r[c]) for c in order]
        lines.append('        ' + sh + ' & '.join(cells) + r'\\')
    lines += [r'        \bottomrule', r'    \end{tabular}', r'\end{table*}']
    (FIGDIR / 'patient_characteristics_table.tex').write_text('\n'.join(lines), encoding='utf-8')
    print(f'  -> {FIGDIR}/patient_characteristics_table.tex (+.csv)')


# ================================================================= ROC ======
def _auc_ci(y, s, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def roc_panel(title, curves, stem):
    fig, ax = plt.subplots(figsize=(SINGLE * 1.25, SINGLE * 1.2))
    ax.plot([0, 1], [0, 1], ls='--', lw=0.6, c='0.6')
    for label, y, s in curves:
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        lo, hi = _auc_ci(y, s)
        ax.plot(fpr, tpr, lw=1.3, label=f'{label} ({auc:.2f}, {lo:.2f}-{hi:.2f})')
    ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
    ax.set_title(f'{title}  (AUROC, 95% CI)', loc='left', fontweight='bold')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    save(fig, stem)


def _ensemble_positive_scores(models, xs, m):
    """Evaluate ensemble member i on imputation i, then average -- the soft
    majority vote the manuscript describes. Falls back to imputation 0 if the
    checkpoint holds a different number of members than there are imputations."""
    probas = []
    for i, mdl in enumerate(models):
        x = np.asarray(xs[i if (len(models) == m and i < m) else 0])
        p = mdl.predict_proba(x)
        probas.append(np.asarray(p))
    p = np.mean(probas, axis=0)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()


def roc_curves():
    for task, title in ROC_TASKS.items():
        try:
            c = load_task(task)
        except Exception as exc:
            print(f'  {task}: prepare_data failed ({exc}); skipped')
            continue
        xs_test, ys_test = c['raw'][4], c['raw'][5]
        xs_pro, ys_pro = c['raw'][6], c['raw'][7]
        y_int = np.asarray(ys_test[0]).ravel()
        y_ext = np.asarray(ys_pro[0]).ravel()

        for cohort, xs, y in (('umm', xs_test, y_int), ('mainz', xs_pro, y_ext)):
            curves = []
            for key, label in ROC_MODELS.items():
                path = MODELS_DIR / key / f'model_{task}.pickle'
                if not path.exists():
                    continue
                try:
                    with open(path, 'rb') as fh:
                        models = pickle.load(fh)
                    curves.append((label, y, _ensemble_positive_scores(models, xs, c['m'])))
                except Exception as exc:
                    print(f'  {key}/{task}/{cohort}: {exc}')
            if curves:
                roc_panel(f'{title} - {cohort.upper()} (n={len(y)})', curves,
                          f'roc_{task}_{cohort}')
            else:
                print(f'  {task}/{cohort}: no usable model checkpoints')


# ================================================================= main =====
def main():
    pub_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    raw = None
    if RAW_SNAPSHOT is not None and Path(RAW_SNAPSHOT).exists():
        raw = pd.read_csv(RAW_SNAPSHOT)
        print(f'Raw snapshot: {RAW_SNAPSHOT} ({len(raw)} rows)')

    print('\n[0/4] cohort')
    cohort = load_task(PRIMARY_TASK)
    n_umm, n_mainz = report_cohort(cohort)

    print('\n[1/4] stage histogram')
    order = ['F0', 'F1', 'F2', 'F3', 'F4']
    umm_lab = mainz_lab = None
    if raw is not None:
        g = grades_from_snapshot(raw)
        if g is not None:
            umm_lab = g[raw.cohort == 'umm'].values
            mainz_lab = g[raw.cohort == 'mainz'].values
    if umm_lab is None:
        umm_lab, mainz_lab = derive_grades()
        order = ['F0/1', 'F2', 'F3', 'F4']
        if umm_lab is not None:
            print('  NOTE: grades reconstructed from the task labels; F0 and F1 are merged.')
    if umm_lab is None:
        print('  skipped -- no grade source available')
    else:
        stage_histograms(umm_lab, mainz_lab, order)

    print('\n[2/4] PCA domain shift + stage structure')
    pca_domain_shift(cohort['X_umm'].mean(axis=0), cohort['X_mainz'].mean(axis=0),
                     umm_lab if order == ['F0/1', 'F2', 'F3', 'F4'] else None,
                     mainz_lab if order == ['F0/1', 'F2', 'F3', 'F4'] else None)

    print('\n[3/4] patient characteristics table')
    characteristics_table(cohort['cols'], cohort['X_umm'], cohort['X_mainz'], raw)

    print('\n[4/4] ROC curves')
    roc_curves()

    print(f'\nDone. All numbers derived from prepare_data(): UMM n={n_umm}, MAINZ n={n_mainz}.')


if __name__ == '__main__':
    os.chdir(SRC_DIR)
    main()


# ---------------------------------------------------------------------------
# OPTIONAL: writing the raw snapshot from inside preprocess.py
#
# Insert this where df_umm and df_pro exist as DataFrames -- the same place the
# cohort figures were hooked in -- but AFTER the eligibility filters, so the
# snapshot covers exactly the analytic cohort and not the raw 654 records:
#
#     import pandas as pd, os
#     os.makedirs('outputs/data_qc', exist_ok=True)
#     _u = df_umm.copy();  _u['cohort'] = 'umm'
#     _m = df_pro.copy();  _m['cohort'] = 'mainz'
#     if 'Micro' in _u.columns:
#         _u['Micro_raw'] = _u['Micro']      # keep F0-F4 before binarisation
#     if 'Micro' in _m.columns:
#         _m['Micro_raw'] = _m['Micro']
#     pd.concat([_u, _m], ignore_index=True).to_csv(
#         'outputs/data_qc/raw_cohort_snapshot.csv', index=False)
#
# Then set RAW_SNAPSHOT at the top of this file. Because it is written during
# the run, it cannot go stale the way ../data/preprocessed_no_mice_data.csv did.
# ---------------------------------------------------------------------------
