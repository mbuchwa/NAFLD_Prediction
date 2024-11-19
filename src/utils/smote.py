from imblearn.over_sampling import SMOTE
import numpy as np


def apply_smote_to_datasets(x_list, y_list, sampling_strategy='auto', random_state=42):
    """
    Applies SMOTE to a list of datasets in the form of lists of arrays.

    Parameters:
    - datasets (list of lists): Each sublist should contain two elements:
                                - X (numpy array): Feature array
                                - y (numpy array): Label array
    - sampling_strategy (str or float): Sampling strategy for SMOTE (default is 'auto')
    - random_state (int): Random state for reproducibility (default is 42)

    Returns:
    - resampled_datasets (list of lists): Each sublist contains resampled X and y arrays
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    resampled_x_list, resampled_y_list = [], []

    for i, (X, y) in enumerate(zip(x_list, y_list)):
        X, y = np.array(X), np.array(y)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        resampled_x_list.append(X_resampled)
        resampled_y_list.append(y_resampled)

        # Print original and resampled class distributions
        print(f"Dataset {i + 1}:")
        print("Original class distribution:", np.bincount(y))
        print("Resampled class distribution:", np.bincount(y_resampled))
        print("-" * 40)

    return resampled_x_list, resampled_y_list