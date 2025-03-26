import pyro
import pyro.distributions as dist
from collections import namedtuple
import shap
from pyro.distributions.util import scalar_like
from pyro.infer.autoguide import init_to_uniform
from pyro.infer.mcmc.hmc import HMC
from pyro.ops.integrator import potential_grad, velocity_verlet
from pyro.util import optional, torch_isnan
from torch import nn
from pyro.infer import MCMC
from pyro.nn import PyroModule, PyroSample
from src.utils.helper_functions import *
import os


def hypertrain_ensemble_mcmc_bnn(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols, classification_type,
                                 shap_selected):
    models = []
    posterior_samples = []
    models_params = []

    if not os.path.isdir('./models/mcmc_bnn'):
        os.mkdir('./models/mcmc_bnn')

    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'training model {idx}')
        best_model, posterior_sample, model_params = hypertrain_mcmc_bnn(X_train, y_train, X_val, y_val,
                                                                         classification_type=classification_type)
        models.append(best_model)
        posterior_samples.append(posterior_sample)
        models_params.append(model_params)

        # Define file path for each model
        model_path = f"./models/mcmc_bnn/model_{classification_type}_{idx}.pth"

        # saving the model
        torch.save(best_model.state_dict(), model_path)

        # saving the mcmc samples
        torch.save(posterior_sample, f'./models/mcmc_bnn/bnn_posterior_samples_{classification_type}_{idx}.pth')

        # saving model params to txt file
        with open(f'./models/mcmc_bnn/model_params_{classification_type}_{idx}.txt', 'w') as txt_f:
            txt_f.write(str(model_params))

    # Ensemble Prediction
    # TODO Check if ensemble function is correct for mcmc_bnn
    ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas = \
        make_ensemble_preds(xs_test, ys_test, models, 'mcmc_bnn')

    plot_accuracies(ensemble_reports, 'Ensemble', name=f'mcmc_bnn', classification_type=classification_type)

    plot_combined_roc_curve(ys_test, ensemble_pred_probas, 'mcmc_bnn', classification_type=classification_type)  # Plot combined ROC

    plot_cm(ensemble_cms[0], name='mcmc_bnn', classification_type=classification_type)

    r, _ = test(y=ys_test[0], y_pred=ensemble_preds[0])
    print('Results on one data set: ', r)

    with open(f'outputs/mcmc_bnn/mcmc_bnn_{classification_type}.csv', 'w') as f:
        for (cm, report) in zip(ensemble_cms, ensemble_reports):
            f.write(str(report))
            f.write(str(cm))

    acc, f1, ppv, tpr = [], [], [], []
    for report in ensemble_reports:
        acc.append(report['accuracy'])
        f1.append(report['macro avg']['f1-score'])
        ppv.append(report['macro avg']['precision'])
        tpr.append(report['macro avg']['recall'])

    performance_string = f'\n Ensemble MCMC_BNN\n' \
                         f'Average ACC: {np.round(np.average(acc) * 100, 2)} | Std ACC: {np.round(np.std(acc) * 100, 2)},\n' \
                         f'Average F1: {np.round(np.average(f1) * 100, 2)} | Std F1: {np.round(np.std(f1) * 100, 2)},\n' \
                         f'Average PPV: {np.round(np.average(ppv) * 100, 2)} | Std PPV: {np.round(np.std(ppv) * 100, 2)},\n' \
                         f'Average TPR: {np.round(np.average(tpr) * 100, 2)} | Std TPR: {np.round(np.std(tpr) * 100, 2)}\n'
    print(performance_string)

    with open(f'outputs/mcmc_bnn/mcmc_bnn_{classification_type}_performance_metrics.txt', 'w') as f:
        f.write(performance_string)
        f.close()

    interpret_mcmc_bnn(xs_train[0], xs_test[0], df_cols, classification_type=classification_type)


def hypertrain_mcmc_bnn(x_train, y_train, x_val, y_val, cv=5, n_iter=5, mcmc_samples=100,
                        classification_type='fibrosis'):
    """
    Hyperparameter tuning and training a PyTorch Lightning model on the provided features (x_train) and labels (y_train) using random search.

    Args:
        x_train (np.array): The features used for training.
        y_train (np.array): The labels used for training.
        x_val (np.array): The features used for validation.
        y_val (np.array): The labels used for validation.
        cv (int): Number of cross-validation folds (default=3).
        n_iter (int): Number of parameter settings that are sampled (default=10).
        mcmc_samples (int): Number of mcmc_samples (default 500).
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'

    Returns:
        MCMC_BNN model: Best trained model found during random search.
    """

    # Merge train and validation data
    x = np.concatenate((x_train, x_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    best_model = None
    best_params = None
    best_score = float('-inf')

    hp_space = {
        'num_layers': np.arange(1, 3),
        'hidden_dim': np.array([32, 64])
    }

    # Initialize cross-validation splitter
    # kf = KFold(n_splits=cv)

    # sklearn KFold does not return same length of fold x and fold y if x.shape[0] % cv != 0 !
    # get the remainder
    # b = x.shape[0] % cv
    #
    # # drop the remainder samples
    # x = x[:-1 * b]
    # y = y[:-1 * b]

    for _ in range(n_iter):
        # Set seed
        np.random.seed(42)

        # Perform cross-validation
        scores = []

        # MCMC BNN is not validated!

        # for train_index, val_index in kf.split(x):
        #     x_train_fold, x_val_fold = x[train_index], x[val_index]
        #     y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Sample hyperparameters from the search space
        sampled_params = {param: np.random.choice(values) for param, values in hp_space.items()}
        sampled_params['input_dim'] = x_train.shape[1]
        sampled_params['out_dim'] = 3 if classification_type == 'three_stage' else 2

        # Train model
        model, posterior_sample, report, cm = train_val_mcmc_bnn_model(torch.tensor(x),
                                                                       torch.tensor(y),
                                                                       torch.tensor(x_val),
                                                                       torch.tensor(y_val),
                                                                       sampled_params,
                                                                       mcmc_samples=mcmc_samples)
        scores.append(report['accuracy'])

        # Calculate average score across folds
        avg_score = np.mean(scores)
        print(f'\n best model has acc: {avg_score}')

        # Update best model if the score is better
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_model_posterior_sample = posterior_sample
            best_params = sampled_params

    return best_model, best_model_posterior_sample, best_params


def train_val_mcmc_bnn_model(x_train, y_train, x_val, y_val, sampled_params, mcmc_samples=100):
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)

    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    model = MCMC_BNN(**sampled_params).to(get_device(i=0))

    pyro.set_rng_seed(42)
    nuts_kernel = NUTS(model, jit_compile=True)
    mcmc = MCMC(nuts_kernel, num_samples=mcmc_samples)
    print('running mcmc ...')
    mcmc.run(x.to(get_device(i=0)), y.to(get_device(i=0)))
    posterior_samples = mcmc.get_samples()

    all_preds = []
    for i in range(len(posterior_samples['layer0.weight'])):
        for j in range(int(len(posterior_samples.items()) / 2)):
            layer_weight = posterior_samples[f'layer{j}.weight'][i]
            layer_bias = posterior_samples[f'layer{j}.bias'][i]
            setattr(model, f'layer{j}.weight', layer_weight)
            setattr(model, f'layer{j}.bias', layer_bias)
        with torch.no_grad():
            mu = model(x_val.to(get_device(i=0)))
        all_preds.append(mu)

    logits_samples = torch.stack(all_preds)
    logits_array = logits_samples.cpu().numpy()
    mean_logits = logits_array.mean(axis=0)
    std_logits = logits_array.std(axis=0)
    class_predictions = np.argmax(mean_logits, axis=1).squeeze()
    report = classification_report(y_val, class_predictions, output_dict=True)
    cm = confusion_matrix(y_val, class_predictions)
    # print(report)
    # print(f'std of logits is：{np.mean(std_logits, axis=0)}')
    return model, posterior_samples, report, cm


def predict_mcmc_bnn_models(data, classification_type='fibrosis'):
    """
    Predictions of the MCMC BNN model for certain data. Models are saved as checkpoint files in a directory.

    Args:
        model_directory: Directory where the PyTorch model checkpoints are stored.
        data: A numpy array to predict on.
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.
    Returns:
        A numpy array of class predictions
    """
    data = torch.tensor(data, dtype=torch.float32, device=get_device(i=0))

    # Get a list of all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir("./models/mcmc_bnn/") if f.endswith('.pth') and 'model' in f and classification_type in f]

    # Load each model checkpoint and make predictions
    y_preds = []
    for checkpoint_file in checkpoint_files:
        # Read the respective hyperparam file
        model_index = checkpoint_file.split('_')[-1].rstrip('.pth')
        with open(f"./models/mcmc_bnn/model_params__{classification_type}_{model_index}.txt", "r") as file:
            # Read the first line
            dict_str = file.readline().strip()

            # Convert the string representation of the dictionary to a Python dictionary
            param_dict = eval(dict_str)
            param_dict['input_dim'] = data.shape[1]
            param_dict['out_dim'] = 2

        model = MCMC_BNN(**param_dict).to(get_device(i=0))

        # Load the respective posterior sample
        model.load_state_dict(torch.load(f'models/mcmc_bnn/{checkpoint_file}'))
        posterior_samples = torch.load(f'models/mcmc_bnn/bnn_posterior_samples_{classification_type}_{model_index}.pth')

        for i in range(len(posterior_samples['layer0.weight'])):
            for j in range(int(len(posterior_samples.items()) / 2)):
                layer_weight = posterior_samples[f'layer{j}.weight'][i]
                layer_bias = posterior_samples[f'layer{j}.bias'][i]
                setattr(model, f'layer{j}.weight', layer_weight)
                setattr(model, f'layer{j}.bias', layer_bias)

        # Make predictions
        with torch.no_grad():
            outputs = model(data)
            y_pred = torch.softmax(outputs.detach().cpu(), dim=1).numpy()  # Assuming output is probability distribution
            y_preds.append(y_pred)

        del model

    # Perform majority voting
    maj_preds = majority_vote(y_preds, rule='soft')  # You need to implement majority_vote function
    indices, _ = get_index_and_proba(maj_preds)  # You need to implement get_index_and_proba function

    return np.array(indices)


def interpret_mcmc_bnn(x_train, x_test, df_cols, classification_type='fibrosis'):
    """
    Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.

    Parameters:
        x_train (array-like): Training data features.
        x_test (array-like): Test data features.
        df_cols (list): List of column names (features).
        classification_type (str): 'fibrosis', 'cirrhosis', 'two_stage' or 'three_stage'.

    Returns:
        None
    """
    explainer = shap.KernelExplainer(model=lambda x: predict_mcmc_bnn_models(x, classification_type=classification_type),
                                     data=shap.sample(x_train, 50), feature_names=df_cols)

    shap_values = explainer.shap_values(x_test)

    f = shap.force_plot(
        explainer.expected_value,
        shap_values,
        x_test,
        feature_names=df_cols,
        show=False)

    shap.save_html(f'outputs/mcmc_bnn/{classification_type}_force_plot.htm', f)
    plt.close()

    fig, ax = plt.subplots()
    shap_values2 = explainer(x_test)
    shap.plots.bar(shap_values2, show=False)

    f = plt.gcf()
    f.savefig(f'outputs/mcmc_bnn/{classification_type}_summary_bar.png', bbox_inches='tight', dpi=300)
    plt.close()

    fig, ax = plt.subplots()

    shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)

    f = plt.gcf()
    f.savefig(f'outputs/mcmc_bnn/{classification_type}_beeswarm.png', bbox_inches='tight', dpi=300)
    plt.close()


def _logaddexp(x, y):
    minval, maxval = (x, y) if x < y else (y, x)
    return (minval - maxval).exp().log1p() + maxval


_TreeInfo = namedtuple(
    "TreeInfo",
    [
        "z_left",
        "r_left",
        "r_left_unscaled",
        "z_left_grads",
        "z_right",
        "r_right",
        "r_right_unscaled",
        "z_right_grads",
        "z_proposal",
        "z_proposal_pe",
        "z_proposal_grads",
        "r_sum",
        "weight",
        "turning",
        "diverging",
        "sum_accept_probs",
        "num_proposals",
    ],
)


# This part is the NUTS kernel which is from The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
# https://arxiv.org/abs/1111.4246
class NUTS(HMC):
    def __init__(
        self,
        model=None,
        potential_fn=None,
        step_size=1,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False,
        use_multinomial_sampling=True,
        transforms=None,
        max_plate_nesting=None,
        jit_compile=False,
        jit_options=None,
        ignore_jit_warnings=False,
        target_accept_prob=0.8,
        max_tree_depth=10,
        init_strategy=init_to_uniform,
    ):
        super().__init__(
            model,
            potential_fn,
            step_size,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            full_mass=full_mass,
            transforms=transforms,
            max_plate_nesting=max_plate_nesting,
            jit_compile=jit_compile,
            jit_options=jit_options,
            ignore_jit_warnings=ignore_jit_warnings,
            target_accept_prob=target_accept_prob,
            init_strategy=init_strategy,
        )
        self.use_multinomial_sampling = use_multinomial_sampling
        self._max_tree_depth = max_tree_depth
        self._max_sliced_energy = 1000

    def _is_turning(self, r_left_unscaled, r_right_unscaled, r_sum):
        left_angle = 0.0
        right_angle = 0.0
        for site_names, value in r_sum.items():
            rho = (value -
                   (r_left_unscaled[site_names] +
                    r_right_unscaled[site_names]) /
                   2)
            left_angle += r_left_unscaled[site_names].dot(rho)
            right_angle += r_right_unscaled[site_names].dot(rho)

        return (left_angle <= 0) or (right_angle <= 0)

    def _build_basetree(
            self,
            z,
            r,
            z_grads,
            log_slice,
            direction,
            energy_current):
        step_size = self.step_size if direction == 1 else -self.step_size
        z_new, r_new, z_grads, potential_energy = velocity_verlet(
            z,
            r,
            self.potential_fn,
            self.mass_matrix_adapter.kinetic_grad,
            step_size,
            z_grads=z_grads,
        )
        r_new_unscaled = self.mass_matrix_adapter.unscale(r_new)
        energy_new = potential_energy + self._kinetic_energy(r_new_unscaled)

        energy_new = (
            scalar_like(energy_new, float("inf"))
            if torch_isnan(energy_new)
            else energy_new
        )
        sliced_energy = energy_new + log_slice
        diverging = sliced_energy > self._max_sliced_energy
        delta_energy = energy_new - energy_current
        accept_prob = (-delta_energy).exp().clamp(max=1.0)

        if self.use_multinomial_sampling:
            tree_weight = -sliced_energy
        else:
            tree_weight = scalar_like(
                sliced_energy, 1.0 if sliced_energy <= 0 else 0.0)

        r_sum = r_new_unscaled
        return _TreeInfo(
            z_new,
            r_new,
            r_new_unscaled,
            z_grads,
            z_new,
            r_new,
            r_new_unscaled,
            z_grads,
            z_new,
            potential_energy,
            z_grads,
            r_sum,
            tree_weight,
            False,
            diverging,
            accept_prob,
            1,
        )

    def _build_tree(
        self, z, r, z_grads, log_slice, direction, tree_depth, energy_current
    ):
        if tree_depth == 0:
            return self._build_basetree(
                z, r, z_grads, log_slice, direction, energy_current
            )

        half_tree = self._build_tree(
            z, r, z_grads, log_slice, direction, tree_depth - 1, energy_current
        )
        z_proposal = half_tree.z_proposal
        z_proposal_pe = half_tree.z_proposal_pe
        z_proposal_grads = half_tree.z_proposal_grads

        if half_tree.turning or half_tree.diverging:
            return half_tree
        if direction == 1:
            z = half_tree.z_right
            r = half_tree.r_right
            z_grads = half_tree.z_right_grads
        else:
            z = half_tree.z_left
            r = half_tree.r_left
            z_grads = half_tree.z_left_grads
        other_half_tree = self._build_tree(
            z, r, z_grads, log_slice, direction, tree_depth - 1, energy_current
        )

        if self.use_multinomial_sampling:
            tree_weight = _logaddexp(half_tree.weight, other_half_tree.weight)
        else:
            tree_weight = half_tree.weight + other_half_tree.weight
        sum_accept_probs = half_tree.sum_accept_probs + other_half_tree.sum_accept_probs
        num_proposals = half_tree.num_proposals + other_half_tree.num_proposals
        r_sum = {
            site_names: half_tree.r_sum[site_names] + other_half_tree.r_sum[site_names]
            for site_names in self.inverse_mass_matrix
        }
        if self.use_multinomial_sampling:
            other_half_tree_prob = (other_half_tree.weight - tree_weight).exp()
        else:
            other_half_tree_prob = (
                other_half_tree.weight / tree_weight
                if tree_weight > 0
                else scalar_like(tree_weight, 0.0)
            )
        is_other_half_tree = pyro.sample(
            "is_other_half_tree", dist.Bernoulli(probs=other_half_tree_prob)
        )

        if is_other_half_tree == 1:
            z_proposal = other_half_tree.z_proposal
            z_proposal_pe = other_half_tree.z_proposal_pe
            z_proposal_grads = other_half_tree.z_proposal_grads

        if direction == 1:
            z_left = half_tree.z_left
            r_left = half_tree.r_left
            r_left_unscaled = half_tree.r_left_unscaled
            z_left_grads = half_tree.z_left_grads
            z_right = other_half_tree.z_right
            r_right = other_half_tree.r_right
            r_right_unscaled = other_half_tree.r_right_unscaled
            z_right_grads = other_half_tree.z_right_grads
        else:
            z_left = other_half_tree.z_left
            r_left = other_half_tree.r_left
            r_left_unscaled = other_half_tree.r_left_unscaled
            z_left_grads = other_half_tree.z_left_grads
            z_right = half_tree.z_right
            r_right = half_tree.r_right
            r_right_unscaled = half_tree.r_right_unscaled
            z_right_grads = half_tree.z_right_grads
        turning = other_half_tree.turning or self._is_turning(
            r_left_unscaled, r_right_unscaled, r_sum
        )
        diverging = other_half_tree.diverging
        return _TreeInfo(
            z_left,
            r_left,
            r_left_unscaled,
            z_left_grads,
            z_right,
            r_right,
            r_right_unscaled,
            z_right_grads,
            z_proposal,
            z_proposal_pe,
            z_proposal_grads,
            r_sum,
            tree_weight,
            turning,
            diverging,
            sum_accept_probs,
            num_proposals,
        )

    def sample(self, params):
        z, potential_energy, z_grads = self._fetch_from_cache()
        if z is None:
            z = params
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
            self._cache(z, potential_energy, z_grads)
        elif len(z) == 0:
            self._t += 1
            self._mean_accept_prob = 1.0
            if self._t > self._warmup_steps:
                self._accept_cnt += 1
            return z
        r, r_unscaled = self._sample_r(name="r_t={}".format(self._t))
        energy_current = self._kinetic_energy(r_unscaled) + potential_energy
        if self.use_multinomial_sampling:
            log_slice = -energy_current
        else:
            slice_exp_term = pyro.sample(
                "slicevar_exp_t={}".format(self._t),
                dist.Exponential(scalar_like(energy_current, 1.0)),
            )
            log_slice = -energy_current - slice_exp_term
        z_left = z_right = z
        r_left = r_right = r
        r_left_unscaled = r_right_unscaled = r_unscaled
        z_left_grads = z_right_grads = z_grads
        accepted = False
        r_sum = r_unscaled
        sum_accept_probs = 0.0
        num_proposals = 0
        tree_weight = scalar_like(
            energy_current, 0.0 if self.use_multinomial_sampling else 1.0
        )
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            tree_depth = 0
            while tree_depth < self._max_tree_depth:
                direction = pyro.sample(
                    "direction_t={}_treedepth={}".format(self._t, tree_depth),
                    dist.Bernoulli(probs=scalar_like(tree_weight, 0.5)),
                )
                direction = int(direction.item())
                if (
                    direction == 1
                ):
                    new_tree = self._build_tree(
                        z_right,
                        r_right,
                        z_right_grads,
                        log_slice,
                        direction,
                        tree_depth,
                        energy_current,
                    )
                    z_right = new_tree.z_right
                    r_right = new_tree.r_right
                    r_right_unscaled = new_tree.r_right_unscaled
                    z_right_grads = new_tree.z_right_grads
                else:
                    new_tree = self._build_tree(
                        z_left,
                        r_left,
                        z_left_grads,
                        log_slice,
                        direction,
                        tree_depth,
                        energy_current,
                    )
                    z_left = new_tree.z_left
                    r_left = new_tree.r_left
                    r_left_unscaled = new_tree.r_left_unscaled
                    z_left_grads = new_tree.z_left_grads
                sum_accept_probs = sum_accept_probs + new_tree.sum_accept_probs
                num_proposals = num_proposals + new_tree.num_proposals
                if new_tree.diverging:
                    if self._t >= self._warmup_steps:
                        self._divergences.append(self._t - self._warmup_steps)
                    break
                if new_tree.turning:
                    break
                tree_depth += 1
                if self.use_multinomial_sampling:
                    new_tree_prob = (new_tree.weight - tree_weight).exp()
                else:
                    new_tree_prob = new_tree.weight / tree_weight
                rand = pyro.sample(
                    "rand_t={}_treedepth={}".format(
                        self._t, tree_depth), dist.Uniform(
                        scalar_like(
                            new_tree_prob, 0.0), scalar_like(
                            new_tree_prob, 1.0)), )
                if rand < new_tree_prob:
                    accepted = True
                    z = new_tree.z_proposal
                    z_grads = new_tree.z_proposal_grads
                    self._cache(z, new_tree.z_proposal_pe, z_grads)
                r_sum = {
                    site_names: r_sum[site_names] + new_tree.r_sum[site_names]
                    for site_names in r_unscaled
                }
                if self._is_turning(
                    r_left_unscaled, r_right_unscaled, r_sum
                ):
                    break
                else:
                    if self.use_multinomial_sampling:
                        tree_weight = _logaddexp(tree_weight, new_tree.weight)
                    else:
                        tree_weight = tree_weight + new_tree.weight
        accept_prob = sum_accept_probs / num_proposals
        self._t += 1
        if self._t > self._warmup_steps:
            n = self._t - self._warmup_steps
            if accepted:
                self._accept_cnt += 1
        else:
            n = self._t
            self._adapter.step(self._t, z, accept_prob, z_grads)
        self._mean_accept_prob += (accept_prob.item() -
                                   self._mean_accept_prob) / n
        return z.copy()


class MCMC_BNN(PyroModule):
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers, prior_scale=1.):
        super().__init__()

        self.activation = nn.ReLU()
        self.num_layers = num_layers

        # Add layers
        for i in range(num_layers):
            if i == 0:
                if num_layers > 1:
                    # First layer
                    layer = PyroModule[nn.Linear](input_dim, hidden_dim)
                    in_features = input_dim
                    out_features = hidden_dim
                else:
                    # First and only layer
                    layer = PyroModule[nn.Linear](input_dim, out_dim)
                    in_features = input_dim
                    out_features = out_dim

            elif i != num_layers-1:
                # Intermediate layers
                layer = PyroModule[nn.Linear](hidden_dim, hidden_dim)
                in_features = hidden_dim
                out_features = hidden_dim

            else:
                # Last layer
                layer = PyroModule[nn.Linear](hidden_dim, out_dim)
                in_features = hidden_dim
                out_features = out_dim

            weight_prior = dist.Normal(0., torch.tensor(prior_scale / (i + 1), device=get_device(i=0)))
            bias_prior = dist.Normal(0., torch.tensor(prior_scale / (i + 1), device=get_device(i=0)))

            layer.weight = PyroSample(weight_prior.expand([out_features, in_features]).to_event(2))
            layer.bias = PyroSample(bias_prior.expand([out_features]).to_event(1))

            setattr(self, f'layer{i}', layer)
            setattr(self, f'layer{i}.weight', layer.weight)
            setattr(self, f'layer{i}.bias', layer.bias)

        self.output = nn.Softmax(dim=1)

    def forward(self, x, y=None):
        for i in range(self.num_layers):
            layer = getattr(self, f'layer{i}')
            x = layer(x)
            if i != self.num_layers - 1:
                x = self.activation(x)
        mu = self.output(x)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=mu), obs=y)
        return mu
