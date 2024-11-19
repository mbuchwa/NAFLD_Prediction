import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from src.preprocess import preprare_data
from src.utils.ger_eng_dict import dict_germ_eng


def analyze_biomarker_distributions(list_umm_train, list_umm_val, list_umm_test,
                                    list_mainz, biomarkers, plot=True, exclude_outliers=True, percentile=99):
    """
    Analyzes the distributions of biomarkers between two cohorts.
    Calculates the Kolmogorov-Smirnov (KS) statistic and Earth Mover's Distance (EMD)
    for each biomarker, and optionally plots the distributions with outlier exclusion.

    Parameters:
        list_umm_train (list of arrays): List of list of arrays for imputed datasets of UMM train cohort, where each
        array contains values for a specific biomarker.
        list_umm_val (list of arrays): List of list of arrays for imputed datasets of UMM val cohort, where each
        array contains values for a specific biomarker.
        list_umm_test (list of arrays): List of list of arrays for imputed datasets of UMM test cohort , where each
        array contains values for a specific biomarker.
        list_mainz (list of arrays): List of list of arrays for imputed datasets of MAINZ cohort 2, where each array
        contains values for a specific biomarker.
        biomarkers (list): List of biomarker names, each corresponding to the index in list1 and list2.
        plot (bool): Whether to plot distributions for each biomarker (default: True).
        exclude_outliers (bool): Whether to exclude outliers based on a percentile cutoff (default: True).
        percentile (int): Percentile threshold for outlier exclusion (default: 99).

    Returns:
        pd.DataFrame: Summary DataFrame with KS statistic and EMD for each biomarker.
    """
    results = []

    # Split plotting into two figures, each with 5 rows and 4 columns (total 20 pairs of plots)
    fig1, axs1 = plt.subplots(5, 4, figsize=(20, 25))

    # Flatten axs for easier indexing and pair each density and histogram plot
    axs_density = axs1.flatten()
    fig3, axs3 = plt.subplots(5, 4, figsize=(20, 25))
    axs_histogram = axs3.flatten()

    for i, biomarker in enumerate(biomarkers):
        # Convert lists to numpy arrays for easier slicing and handling
        arr1 = np.concatenate(
            [np.array(list_umm_train[0])[:, i], np.array(list_umm_val[0])[:, i], np.array(list_umm_test[0])[:, i]])
        arr2 = np.array(list_mainz[0])[:, i]

        # Replace inf values with NaN for both arrays
        arr1 = np.where(np.isinf(arr1), np.nan, arr1)
        arr2 = np.where(np.isinf(arr2), np.nan, arr2)

        # Exclude outliers based on the specified percentile
        if exclude_outliers:
            threshold1 = np.percentile(arr1, percentile)
            threshold2 = np.percentile(arr2, percentile)
            arr1 = arr1[arr1 <= threshold1]
            arr2 = arr2[arr2 <= threshold2]

        # Calculate Kolmogorov-Smirnov (KS) statistic
        ks_stat, ks_p_value = ks_2samp(arr1[~np.isnan(arr1)], arr2[~np.isnan(arr2)])

        # Calculate Earth Mover's Distance (EMD)
        emd = wasserstein_distance(arr1[~np.isnan(arr1)], arr2[~np.isnan(arr2)])

        # Append results
        results.append({
            "Biomarker": biomarker,
            "KS_Statistic": ks_stat,
            "KS_p_value": ks_p_value,
            "EMD": emd
        })

        # Plot density plot
        if plot:
            ax_density = axs_density[i]  # Select density plot subplot
            sns.kdeplot(arr1[~np.isnan(arr1)], shade=True, color="blue", label="UMM cohort", ax=ax_density)
            sns.kdeplot(arr2[~np.isnan(arr2)], shade=True, color="orange", label="MAINZ cohort", ax=ax_density)
            ax_density.set_title(f"Density Plot for {biomarker}")
            ax_density.set_xlabel(biomarker)
            ax_density.set_ylabel("Density")
            ax_density.legend()

            # Plot histogram
            ax_hist = axs_histogram[i]  # Select histogram subplot
            ax_hist.hist(arr1[~np.isnan(arr1)], bins=30, color='blue', alpha=0.5, label="UMM cohort")
            ax_hist.hist(arr2[~np.isnan(arr2)], bins=30, color='orange', alpha=0.5, label="MAINZ cohort")
            ax_hist.set_title(f"Histogram for {biomarker}")
            ax_hist.set_xlabel(biomarker)
            ax_hist.set_ylabel("Frequency")
            ax_hist.legend()

    # Adjust layout for better visibility
    fig1.tight_layout()
    fig3.tight_layout()

    # Show plots
    plt.show()  # Displays all figures

    # Convert results to a DataFrame for easy viewing and saving
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == '__main__':
    classification_type = 'two_stage'
    shap_selected = False
    scaling = False
    select_patients = False

    assert classification_type in ['cirrhosis', 'fibrosis', 'three_stage', 'two_stage']

    xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols = preprare_data(classification_type,
                                                                                                  shap_selected,
                                                                                                  scaling,
                                                                                                  select_patients=select_patients)


    df_cols = [dict_germ_eng[biomarker] for biomarker in df_cols]

    # Assuming df1 and df2 are your DataFrames for UMM cohort and MAINZ cohort
    results_df = analyze_biomarker_distributions(xs_train, xs_val, xs_test, xs_pro, df_cols)

    # Display or save the results
    print(results_df)