"""
make_publication_figures.py
===========================
Regenerates SHAP analyses and confusion matrices in journal-ready quality and
exports the numeric SHAP values that the manuscript tables are built from.

Place in:  src/            Run from:  src/  ->  python make_publication_figures.py

Why this replaces the existing `interpret()` route
--------------------------------------------------
1. `interpret()` explains `predict_model`, which returns the *hard class label*
   (0/1). SHAP is then computed on a step function, which is far less
   informative and produces attributions that are hard to defend in review.
   Here the ensemble's *predicted probability of the positive class* is
   explained instead.
2. `interpret()` uses KernelExplainer with a 50-row background sample - a slow
   approximation. For LightGBM / XGBoost / Random Forest, TreeExplainer gives
   exact Shapley values in seconds.
3. `interpret()` never stores the numeric SHAP values, so the mean-|SHAP| table
   in the manuscript cannot be reproduced from saved artefacts. This script
   writes them to CSV.
4. SHAP is computed for every ensemble member separately, so the table can
   report mean +/- SD across the m models, as the manuscript claims.

Outputs (under outputs/figures/)
    shap_values_<model>_<task>.csv     per-feature mean|SHAP| per ensemble member
    shap_summary_<model>_<task>.pdf    beeswarm (vector)
    shap_bar_<model>_<task>.pdf        mean|SHAP| with SD across ensemble
    confusion_<model>_<task>_<split>.pdf
    shap_top5_table.tex                LaTeX table of the top-5 features
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

warnings.filterwarnings('ignore')

FIGDIR = Path('outputs/figures')
# Journal figure widths (mm -> inch). Nature: 89 single, 183 double column.
SINGLE, DOUBLE = 89 / 25.4, 183 / 25.4

TASKS = ['fibrosis', 'two_stage', 'cirrhosis', 'three_stage']
TASK_LABEL = {'fibrosis': 'Moderate fibrosis', 'two_stage': 'Severe fibrosis',
              'cirrhosis': 'Cirrhosis', 'three_stage': 'Three-stage'}
TREE_MODELS = ['light_gbm', 'xgb', 'rf']          # exact TreeExplainer
MODEL_LABEL = {'light_gbm': 'LightGBM', 'xgb': 'XGBoost', 'rf': 'Random Forest',
               'svm': 'SVM'}
TOP_N = 5

# Colour-blind-safe diverging map (blue -> grey -> red), avoids the default rainbow
CMAP = LinearSegmentedColormap.from_list('cb', ['#2166AC', '#B8B8B8', '#B2182B'])


def pub_style():
    """Rc parameters matching Nature / BMJ figure guidelines."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,      # editable text in Illustrator / journal production
        'ps.fonttype': 42,
    })


def save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / f'{stem}.{ext}')
    plt.close(fig)
    print(f'  -> {FIGDIR}/{stem}.pdf (+.png)')


# ---------------------------------------------------------------- SHAP -----
def shap_per_member(models, x_bg, x_explain, positive_class=1):
    """Exact SHAP values per ensemble member for the positive-class probability."""
    import shap
    per_member = []
    for mdl in models:
        try:
            expl = shap.TreeExplainer(mdl)
            sv = expl.shap_values(x_explain)
        except Exception as exc:
            print(f'    TreeExplainer failed ({exc}); falling back to KernelExplainer')
            expl = shap.KernelExplainer(lambda d: mdl.predict_proba(d)[:, positive_class],
                                        shap.sample(x_bg, 50))
            sv = expl.shap_values(x_explain, silent=True)
        sv = np.asarray(sv)
        # normalise shape to (n_samples, n_features) for the positive class
        if sv.ndim == 3:
            sv = sv[:, :, positive_class] if sv.shape[-1] > positive_class else sv[..., 0]
        elif isinstance(sv, list):
            sv = np.asarray(sv[positive_class])
        per_member.append(sv)
    return per_member


def beeswarm(sv, x, features, title, stem, top_n=12):
    """Custom beeswarm - cleaner and more controllable than shap.summary_plot."""
    order = np.argsort(np.abs(sv).mean(0))[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 0.19 * len(order) + 0.9))

    for row, fi in enumerate(order):
        vals, feat = sv[:, fi], x[:, fi]
        finite = np.isfinite(feat)
        lo, hi = (np.min(feat[finite]), np.max(feat[finite])) if finite.sum() > 1 else (0.0, 0.0)
        c = (feat - lo) / (hi - lo) if hi > lo else np.full_like(feat, 0.5, dtype=float)
        # vertical jitter proportional to local density
        nb = 40
        edges = np.linspace(vals.min(), vals.max() + 1e-12, nb + 1)
        idx = np.clip(np.digitize(vals, edges) - 1, 0, nb - 1)
        y = np.zeros_like(vals, dtype=float)
        for b in range(nb):
            m = idx == b
            k = m.sum()
            if k:
                spread = min(0.36, 0.045 * np.sqrt(k))
                y[m] = np.linspace(-spread, spread, k) if k > 1 else 0.0
        ax.scatter(vals, row + y, c=c, cmap=CMAP, s=5, linewidths=0,
                   alpha=0.85, rasterized=True, vmin=0, vmax=1)

    ax.axvline(0, color='0.75', lw=0.5, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([features[i] for i in order])
    ax.set_xlabel('SHAP value (impact on predicted probability)')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

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


# ------------------------------------------------------------------ main ---
def main():
    pub_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    from src.preprocess import prepare_data
    from src.utils.ger_eng_dict import dict_germ_eng
    from sklearn.metrics import confusion_matrix

    table_rows = []
    for task in TASKS:
        print(f'\n=== {task} ===')
        (xs_train, _, _, _, xs_test, ys_test,
         xs_pro, ys_pro, df_cols_de) = prepare_data(task, False, False)
        feats = [dict_germ_eng.get(c, c) for c in df_cols_de]
        x_te, y_te = np.asarray(xs_test[0]), np.asarray(ys_test[0])
        x_pr, y_pr = np.asarray(xs_pro[0]), np.asarray(ys_pro[0])
        n_cls = 3 if task == 'three_stage' else 2
        classes = ['F0-1', 'F2-3', 'F4'] if n_cls == 3 else ['Negative', 'Positive']

        for model in TREE_MODELS:
            path = Path(f'models/{model}/model_{task}.pickle')
            if not path.exists():
                print(f'  {model}: no checkpoint - skipped')
                continue
            models = pickle.load(open(path, 'rb'))
            print(f'  {model}: {len(models)} ensemble members')

            # ---- SHAP per ensemble member -> mean +/- SD --------------------
            per = shap_per_member(models, np.asarray(xs_train[0]), x_te)
            abs_means = np.vstack([np.abs(s).mean(0) for s in per])   # (m, n_feat)
            mean_abs, sd_abs = abs_means.mean(0), abs_means.std(0, ddof=1)

            pd.DataFrame(abs_means, columns=feats).assign(
                ensemble_member=range(len(per))).to_csv(
                FIGDIR / f'shap_values_{model}_{task}.csv', index=False)

            lbl = MODEL_LABEL.get(model, model)
            beeswarm(per[0], x_te, feats, f'{TASK_LABEL[task]} - {lbl}',
                     f'shap_summary_{model}_{task}')
            bar_plot(mean_abs, sd_abs, feats, f'{TASK_LABEL[task]} - {lbl}',
                     f'shap_bar_{model}_{task}')

            for i in np.argsort(mean_abs)[::-1][:TOP_N]:
                table_rows.append(dict(task=task, model=lbl, feature=feats[i],
                                       mean_abs_shap=mean_abs[i], sd=sd_abs[i]))

            # ---- confusion matrices ----------------------------------------
            for x, y, split in [(x_te, y_te, 'UMM'), (x_pr, y_pr, 'MAINZ')]:
                proba = np.mean([m.predict_proba(x) for m in models], axis=0)
                pred = proba.argmax(1)
                confusion_plot(confusion_matrix(y, pred, labels=range(n_cls)), classes,
                               f'{TASK_LABEL[task]} - {lbl} ({split})',
                               f'confusion_{model}_{task}_{split.lower()}')

    # ---- top-5 LaTeX table -------------------------------------------------
    if table_rows:
        df = pd.DataFrame(table_rows)
        df.to_csv(FIGDIR / 'shap_top5.csv', index=False)
        lines = [r'\begin{table*}[htbp]', r'    \centering',
                 r'    \caption{\small{Five most influential biomarkers per task and model,',
                 r'    quantified as the mean absolute SHAP value on the held-out UMM test partition.',
                 r'    Values are mean $\pm$ standard deviation across the $m=10$ ensemble members.}}',
                 r'    \label{tab:shap_top5}', r'    \begin{tabular}{llc}', r'        \toprule',
                 r'        \textbf{Task} & \textbf{Biomarker} & \textbf{mean $|$SHAP$|$}\\',
                 r'        \midrule']
        for task in TASKS:
            sub = df[df.task == task]
            if sub.empty:
                continue
            best = sub.model.iloc[0]
            sub = sub[sub.model == best]
            lines.append(f'        \\multicolumn{{3}}{{l}}{{\\textit{{{TASK_LABEL[task]}}} '
                         f'({best})}}\\\\')
            for _, r in sub.iterrows():
                lines.append(f'        & {r.feature} & {r.mean_abs_shap:.3f} $\\pm$ {r.sd:.3f}\\\\')
            lines.append(r'        \midrule')
        lines[-1] = r'        \bottomrule'
        lines += [r'    \end{tabular}', r'\end{table*}']
        (FIGDIR / 'shap_top5_table.tex').write_text('\n'.join(lines))
        print(f'\nLaTeX table -> {FIGDIR}/shap_top5_table.tex')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()
