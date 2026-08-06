"""
neural_loaders.py
=================
Loads the VI-BNN and GANDALF ensembles and wraps them so they expose
`predict_proba(x)` like the sklearn models. That lets recompute_tables.py,
recompute_three_stage.py and check_table_figure_consistency.py treat all models
identically.

Place in:  src/neural_loaders.py

    from src.neural_loaders import load_any_ensemble
    models = load_any_ensemble('vi_bnn', 'cirrhosis')      # -> list of wrappers
    models = load_any_ensemble('gandalf', 'cirrhosis')

--------------------------------------------------------------------------
WHY THE STANDARD LOADER MISSED THESE
--------------------------------------------------------------------------
The tree models pickle the whole ensemble into one file, `model_<task>.pickle`.
The two neural models save one artefact PER ensemble member:

    vi_bnn    model_<task>_<idx>.pth        torch state_dict (a file)
              model_params_<task>_<idx>.txt architecture, repr() of a dict
    gandalf   model_<task>_<idx>.pth        a DIRECTORY in pytorch_tabular format
              model_params_<task>_<idx>.txt hyperparameters
              df_cols.txt                   column order used at training time

--------------------------------------------------------------------------
TWO BUGS THIS FIXES
--------------------------------------------------------------------------
1. ENSEMBLE ORDER. Both evaluate_ensemble_* functions in the repo build their
   model list with a plain `os.listdir(...)` comprehension. os.listdir returns
   entries in arbitrary filesystem order, so member i was not imputation i --
   the pairing the manuscript describes was broken for these two models. Here
   the index is parsed out of the filename and the list is sorted by it.

2. STOCHASTIC PREDICTIONS. VI_BNN samples its weights on every forward pass, so
   a single call gives a different answer each time. The repo predicts once.
   VIBNNWrapper averages the softmax over N_POSTERIOR_SAMPLES passes under a
   fixed torch seed, which is both the correct posterior predictive and
   reproducible. Raise N_POSTERIOR_SAMPLES if the AUROC still moves between
   runs; 200 is usually enough for a model this small.

--------------------------------------------------------------------------
A NOTE ON THE VI-BNN ARCHITECTURE
--------------------------------------------------------------------------
The saved state_dict for fibrosis/0 holds exactly four tensors:

    hidden_layers.0.0.w_mean  (2, 20)     w_std, b_mean, b_std alike

That is a single Bayesian linear layer mapping 20 inputs directly to 2 classes.
With num_layers=1 the sampled hidden_dim=32 never materialises, so this model is
a Bayesian logistic regression, not a multi-layer network. Worth knowing before
describing it as a "Bayesian neural network" in the manuscript -- and it also
explains why VI-BNN tracks the SVM so closely in the results.
"""

import ast
import os
import re
from pathlib import Path

import numpy as np

N_POSTERIOR_SAMPLES = 200     # forward passes averaged per VI-BNN prediction
TORCH_SEED = 42


# --------------------------------------------------------------- helpers ---
def _member_files(model_dir, task, suffix='.pth'):
    """All checkpoints for one task, sorted by ensemble index.

    Matches `model_<task>_<idx><suffix>` exactly, so 'two_stage' cannot pick up
    'three_stage' files and the params .txt files are not mistaken for models.
    """
    d = Path(model_dir)
    if not d.is_dir():
        return []
    pat = re.compile(rf'^model_{re.escape(task)}_(\d+){re.escape(suffix)}$')
    hits = []
    for entry in os.listdir(d):
        m = pat.match(entry)
        if m:
            hits.append((int(m.group(1)), d / entry))
    return [p for _, p in sorted(hits)]


def _read_params(model_dir, task, idx):
    """Parse model_params_<task>_<idx>.txt.

    The file is a repr() of a dict containing numpy scalars, e.g.
    "{'lr': np.float64(0.01), 'num_layers': np.int64(1)}". ast.literal_eval
    cannot parse the np.* calls, so they are unwrapped textually first --
    safer than eval(), which is what the repo uses.
    """
    p = Path(model_dir) / f'model_params_{task}_{idx}.txt'
    if not p.exists():
        raise FileNotFoundError(p)
    s = p.read_text(encoding='utf-8').strip()
    s = re.sub(r'np\.\w+\(([^()]*)\)', r'\1', s)
    return ast.literal_eval(s)


def _softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------- VI-BNN ----
class VIBNNWrapper:
    """sklearn-like wrapper around a VI_BNN checkpoint."""

    def __init__(self, model, device, n_samples=N_POSTERIOR_SAMPLES, seed=TORCH_SEED):
        self.model, self.device = model, device
        self.n_samples, self.seed = n_samples, seed

    def predict_proba(self, x):
        import torch
        xt = torch.tensor(np.asarray(x, dtype=np.float32)).to(self.device)
        torch.manual_seed(self.seed)
        acc = None
        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(xt)
                p = torch.softmax(out, dim=1).detach().cpu().numpy()
                acc = p if acc is None else acc + p
        return acc / self.n_samples

    def predict(self, x):
        return self.predict_proba(x).argmax(1)


def load_vi_bnn(task, model_dir='models/vi_bnn', n_samples=N_POSTERIOR_SAMPLES):
    import torch
    try:
        from src.utils.networks import VI_BNN
        from src.utils.helper_functions import get_device
    except ImportError:
        from utils.networks import VI_BNN
        from utils.helper_functions import get_device

    device = get_device(i=0)
    files = _member_files(model_dir, task)
    if not files:
        raise FileNotFoundError(f'no model_{task}_<idx>.pth in {model_dir}')

    models = []
    for path in files:
        idx = int(re.search(r'_(\d+)\.pth$', path.name).group(1))
        params = _read_params(model_dir, task, idx)
        net = VI_BNN(**params, prior_var=1.0).to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        models.append(VIBNNWrapper(net, device, n_samples))
    print(f'  vi_bnn/{task}: {len(models)} members, '
          f'{n_samples} posterior samples per prediction')
    return models


class TorchNetWrapper:
    """Deterministic torch model -> softmax probabilities."""

    def __init__(self, model, device):
        self.model, self.device = model, device

    def predict_proba(self, x):
        import torch
        xt = torch.tensor(np.asarray(x, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(xt)
        return torch.softmax(out, dim=1).detach().cpu().numpy()

    def predict(self, x):
        return self.predict_proba(x).argmax(1)


def _find_net_class(candidates):
    """Locate the network class in src/utils/networks.py by name."""
    try:
        from src.utils import networks
    except ImportError:
        from utils import networks
    for name in candidates:
        cls = getattr(networks, name, None)
        if cls is not None:
            return cls, name
    available = [n for n in dir(networks) if n[0].isupper()]
    raise ImportError(f'none of {candidates} found in networks.py; '
                      f'available classes: {available}')


def load_torch_ensemble(task, model_dir, class_names, extra_params=None):
    """Generic loader for the torch/Lightning ensembles (ffn, tab_transformer).

    Both save one state_dict per member plus a params .txt, but their parameter
    sets differ -- ffn writes input_dim/num_classes, tab_transformer writes
    out_dim and no input_dim. Rather than hard-coding either, the params are
    filtered against the constructor signature and anything unused is reported.
    'lr' is kept if the class accepts it: these are LightningModules, which
    commonly take the learning rate in __init__.
    """
    import inspect
    import torch
    try:
        from src.utils.helper_functions import get_device
    except ImportError:
        from utils.helper_functions import get_device

    cls, cls_name = _find_net_class(class_names)
    device = get_device(i=0)
    files = _member_files(model_dir, task)
    if not files:
        raise FileNotFoundError(f'no model_{task}_<idx>.pth in {model_dir}')

    sig = set(inspect.signature(cls.__init__).parameters) - {'self'}
    models, dropped = [], set()
    for path in files:
        idx = int(re.search(r'_(\d+)\.pth$', path.name).group(1))
        params = dict(_read_params(model_dir, task, idx))
        if extra_params:
            params.update(extra_params)
        dropped |= set(params) - sig
        params = {k: v for k, v in params.items() if k in sig}
        missing = {p for p in sig if p not in params}

        try:
            net = cls(**params)
        except TypeError as exc:
            raise TypeError(
                f'{cls_name}(**{sorted(params)}) failed: {exc}\n'
                f'  constructor takes: {sorted(sig)}\n'
                f'  params file has:   {sorted(_read_params(model_dir, task, idx))}\n'
                f'  unset:             {sorted(missing)}') from None

        net = net.to(device)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']          # Lightning checkpoint wrapper
        net.load_state_dict(state)
        net.eval()
        models.append(TorchNetWrapper(net, device))

    note = f', ignored {sorted(dropped)}' if dropped else ''
    print(f'  {Path(model_dir).name}/{task}: {len(models)} members '
          f'({cls_name}, deterministic{note})')
    return models


def load_ffn(task, model_dir='models/ffn'):
    """MLP. networks.py calls the class NeuralNetwork."""
    return load_torch_ensemble(task, model_dir,
                               ('NeuralNetwork', 'FFN', 'MLP', 'FeedForward'))


def load_tab_transformer(task, model_dir='models/tab_transformer'):
    """TabTransformer: a torch state_dict per member, NOT the pytorch_tabular
    directory format gandalf uses -- gandalf writes directories (config.yml,
    model.ckpt, ...), tab_transformer ~15 MB .pth files.

    The params file has no input_dim, so it is taken from <task>_df_cols.txt if
    the constructor asks for one.
    """
    import inspect
    try:
        from src.utils import networks
    except ImportError:
        from utils import networks
    extra = {}
    try:
        cls, _ = _find_net_class(('PLTabTransformer', 'TabTransformer'))
        sig = set(inspect.signature(cls.__init__).parameters)
        for key in ('input_dim', 'num_features', 'n_features', 'df_cols', 'columns'):
            if key in sig:
                cols = _read_df_cols(model_dir, task)
                extra[key] = cols if key in ('df_cols', 'columns') else len(cols)
                break
    except (ImportError, FileNotFoundError):
        pass
    return load_torch_ensemble(task, model_dir,
                               ('PLTabTransformer', 'TabTransformer',
                                'TabTransformerModel'), extra_params=extra)


# -------------------------------------------------------------- GANDALF ----
class GandalfWrapper:
    """sklearn-like wrapper around a pytorch_tabular TabularModel."""

    def __init__(self, model, df_cols):
        self.model, self.df_cols = model, list(df_cols)

    def predict_proba(self, x):
        import pandas as pd
        x = np.asarray(x)
        if x.shape[1] != len(self.df_cols):
            raise ValueError(f'GANDALF expects {len(self.df_cols)} columns, got {x.shape[1]}')
        df = pd.DataFrame(x, columns=self.df_cols)
        try:
            out = self.model.predict(df)
        except Exception:
            df = df.copy()
            df['target'] = 0            # some versions require the target column
            out = self.model.predict(df)
        # Column naming differs between pytorch_tabular versions:
        #   older: '0_probability', '1_probability'
        #   1.1.x: 'target_0_probability', 'target_1_probability'
        # plus a '<target>_prediction' column in both. Capture the trailing class
        # index and sort by it, so neither the prefix nor the column order matters.
        # (The commented-out helpers in gandalf.py hard-code the older names and
        # would fail on this version too.)
        pat = re.compile(r'^(?:.*_)?(\d+)_probability$')
        hits = [(int(m.group(1)), c) for c in map(str, out.columns)
                if (m := pat.match(c))]
        if not hits:
            raise ValueError(f'no <class>_probability columns in prediction; '
                             f'got {list(out.columns)}')
        cols = [c for _, c in sorted(hits)]
        return out[cols].to_numpy(dtype=float)

    def predict(self, x):
        return self.predict_proba(x).argmax(1)


def _read_df_cols(model_dir, task):
    """Column order. gandalf writes one df_cols.txt, tab_transformer one per task."""
    d = Path(model_dir)
    for name in (f'{task}_df_cols.txt', 'df_cols.txt'):
        p = d / name
        if p.exists():
            return ast.literal_eval(p.read_text(encoding='utf-8').strip())
    raise FileNotFoundError(f'neither {task}_df_cols.txt nor df_cols.txt in {d} — '
                            f'needed for the column order')


def load_pytorch_tabular(task, model_dir):
    """Any pytorch_tabular ensemble (gandalf, tab_transformer).

    The checkpoints are directories named model_<task>_<idx>.pth, so the same
    index-sorted discovery as for the .pth files applies.
    """
    from pytorch_tabular import TabularModel

    df_cols = _read_df_cols(model_dir, task)
    files = _member_files(model_dir, task)
    if not files:
        raise FileNotFoundError(f'no model_{task}_<idx>.pth in {model_dir}')
    models = [GandalfWrapper(TabularModel.load_model(str(p)), df_cols) for p in files]
    print(f'  {Path(model_dir).name}/{task}: {len(models)} members, '
          f'{len(df_cols)} columns')
    return models


def load_gandalf(task, model_dir='models/gandalf'):
    return load_pytorch_tabular(task, model_dir)


# ------------------------------------------------------------- dispatch ----
LOADERS = {'vi_bnn': load_vi_bnn,
           'gandalf': load_gandalf,
           'tab_transformer': load_tab_transformer,
           'ffn': load_ffn}


def load_any_ensemble(name, task, model_dir=None):
    """Load a neural ensemble by directory name. Returns a list of wrappers."""
    if name not in LOADERS:
        raise KeyError(f'no loader for {name!r}; known: {sorted(LOADERS)}')
    return LOADERS[name](task, model_dir or f'models/{name}')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    for name in ('vi_bnn', 'gandalf', 'tab_transformer', 'ffn'):
        for task in ('fibrosis', 'two_stage', 'cirrhosis', 'three_stage'):
            try:
                m = load_any_ensemble(name, task)
                print(f'  OK  {name}/{task}: {len(m)} members')
            except Exception as exc:
                print(f'  --  {name}/{task}: {type(exc).__name__}: {exc}')