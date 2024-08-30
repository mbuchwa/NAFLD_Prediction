import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import pandas as pd


def select_oldest_patients(df1, df2):
    """
    Select the oldest patients from df2 based on the number of patients in df1.

    Parameters:
    df1 (pd.DataFrame): DataFrame containing data for dataset 1.
    df2 (pd.DataFrame): DataFrame containing data for dataset 2.

    Returns:
    pd.DataFrame: DataFrame containing the oldest patients from df2, with the same number of patients as in df1.
    """

    # Calculate the number of patients in df1
    num_patients_in_df1 = len(df1)

    # Sort df2 by age in descending order
    df2_sorted = df2.sort_values(by='Age', ascending=False)

    # Select the same number of oldest patients as in df1
    oldest_patients_df = df2_sorted.head(num_patients_in_df1)

    return oldest_patients_df


def select_highest_platelet_patients(df1, df2):
    """
    Select the oldest patients from df2 based on the number of patients in df1.

    Parameters:
    df1 (pd.DataFrame): DataFrame containing data for dataset 1.
    df2 (pd.DataFrame): DataFrame containing data for dataset 2.

    Returns:
    pd.DataFrame: DataFrame containing the oldest patients from df2, with the same number of patients as in df1.
    """

    # Calculate the number of patients in df1
    num_patients_in_df1 = len(df1)

    # Sort df2 by age in descending order
    df2_sorted = df2.sort_values(by='Thrombozyten (Mrd/l)', ascending=False)

    # Select the same number of oldest patients as in df1
    oldest_patients_df = df2_sorted.head(num_patients_in_df1)

    return oldest_patients_df


def select_closest_patients(df1, df2):
    """
    Select the closest patients from df2 to those in df1 based on their features.

    Parameters:
    df1 (pd.DataFrame): DataFrame containing data for dataset 1.
    df2 (pd.DataFrame): DataFrame containing data for dataset 2.

    Returns:
    list: List of "ID" values from df2 that are closest to the patients in df1.
    """

    # Create copies of the original dataframes to preserve column names
    df2_original = df2.copy()

    # Rename columns to remove special characters and spaces for processing
    df1.columns = df1.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(' ', '')
    df2.columns = df2.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(' ', '')

    df1 = df1[['Age', 'Thrombozyten_Mrd_l_', 'ASAT_U_I_', 'ALAT_U_I_']]
    df2 = df2[['Age', 'Thrombozyten_Mrd_l_', 'ASAT_U_I_', 'ALAT_U_I_']]

    # Combine the datasets for clustering
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(combined_df)

    # Perform PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(standardized_data)

    # Extract PCA components for Dataset 1 and Dataset 2
    reduced_data_df1 = reduced_data[:len(df1), :]
    reduced_data_df2 = reduced_data[len(df1):, :]

    """Use K-Nearest Neighbor approach"""
    df2_scaled = scaler.transform(df2)
    df1_scaled = scaler.transform(df1)

    # Use NearestNeighbors to find the closest patients
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(df2_scaled)
    distances, indices = nbrs.kneighbors(df1_scaled)

    # Determine the common range for x and y axes
    x_min = min(reduced_data[:, 0]) - 1
    x_max = max(reduced_data[:, 0]) + 1
    y_min = min(reduced_data[:, 1]) - 1
    y_max = max(reduced_data[:, 1]) + 1

    # Plot the PCA-reduced data
    plt.figure(figsize=(18, 6))

    # Plot Dataset 1 patients
    plt.subplot(1, 3, 1)
    plt.scatter(
        reduced_data_df1[:, 0],
        reduced_data_df1[:, 1],
        c='blue',
        marker='o',
        alpha=0.7,
        edgecolors='k',
        s=100,
        label='UMM Test Dataset'
    )
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('UMM Test Dataset')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)

    # Plot Dataset 2 patients
    plt.subplot(1, 3, 2)
    plt.scatter(
        reduced_data_df2[:, 0],
        reduced_data_df2[:, 1],
        c='grey',
        marker='o',
        alpha=0.5,
        edgecolors='k',
        s=100,
        label='Mainz Data'
    )
    plt.scatter(
        reduced_data_df2[indices.squeeze(), 0],
        reduced_data_df2[indices.squeeze(), 1],
        c='red',
        marker='^',
        alpha=0.7,
        edgecolors='k',
        s=100,
        label='Closest to UMM Test Dataset'
    )
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Mainz Data')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)

    # Plot Dataset 3 patients
    plt.subplot(1, 3, 3)
    plt.scatter(
        reduced_data_df2[:, 0],
        reduced_data_df2[:, 1],
        c='grey',
        marker='o',
        alpha=0.5,
        edgecolors='k',
        s=100,
        label='Mainz Data'
    )
    plt.scatter(
        reduced_data_df2[indices.squeeze(), 0],
        reduced_data_df2[indices.squeeze(), 1],
        c='red',
        marker='^',
        alpha=0.7,
        edgecolors='k',
        s=100,
        label='Closest to UMM Test Dataset'
    )
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Mainz Data')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Retrieve the original df for the selected patients from Dataset 2
    closest_patients_df = df2_original.iloc[indices.squeeze()]

    return closest_patients_df


# def select_closest_patients_from_arrays(xs_test, xs_pro, ys_test, ys_pro):
#     """
#     Select the closest patients from arr2 to those in arr1 based on their features
#     and plot the clusters.
#
#     Parameters:
#     xs_test (np.ndarray): Array containing data for dataset 1 (e.g., 52 patients).
#     xs_pro (np.ndarray): Array containing data for dataset 2 (e.g., 284 patients).
#     ys_test (np.ndarray): Array containing target for dataset 1 (e.g., 52 patients).
#     ys_pro (np.ndarray): Array containing target for dataset 2 (e.g., 284 patients).
#
#     Returns:
#     list: List of indices from arr2 that are closest to the patients in arr1.
#     """
#     # Take first element of both xs_test and xs_pro
#     arr1 = xs_test[0]
#     arr2 = xs_pro[0]
#
#     # Combine the arrays for processing
#     combined_arr = np.vstack((arr1, arr2))
#
#     # Standardize the data
#     scaler = StandardScaler()
#     standardized_data = scaler.fit_transform(combined_arr)
#
#     # Perform PCA
#     pca = PCA(n_components=2)
#     reduced_data = pca.fit_transform(standardized_data)
#
#     # Extract PCA components for Dataset 1 and Dataset 2
#     reduced_data_arr1 = reduced_data[:len(arr1), :]
#     reduced_data_arr2 = reduced_data[len(arr1):, :]
#
#     # Compute pairwise distances between patients in arr1 and arr2
#     distances = cdist(reduced_data_arr1, reduced_data_arr2, metric='euclidean')
#
#     # Select closest patients from arr2 to those in arr1
#     closest_indices = np.argmin(distances, axis=1)
#
#     """Use K-Nearest Neighbor approach"""
#     df1_scaled = scaler.transform(arr1)
#     df2_scaled = scaler.transform(arr2)
#
#     # Use NearestNeighbors to find the closest patients
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(df2_scaled)
#     distances, indices = nbrs.kneighbors(df1_scaled)
#
#     # Determine the common range for x and y axes
#     x_min = min(reduced_data[:, 0]) - 1
#     x_max = max(reduced_data[:, 0]) + 1
#     y_min = min(reduced_data[:, 1]) - 1
#     y_max = max(reduced_data[:, 1]) + 1
#
#     # Plot the PCA-reduced data
#     plt.figure(figsize=(12, 6))
#
#     # Plot Dataset 1 patients
#     plt.subplot(1, 2, 1)
#     plt.scatter(
#         reduced_data_arr1[:, 0],
#         reduced_data_arr1[:, 1],
#         c='blue',
#         marker='o',
#         alpha=0.7,
#         edgecolors='k',
#         s=100,
#         label='UMM Test Dataset'
#     )
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.title('UMM Test Dataset')
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
#     plt.legend()
#     plt.grid(True)
#
#     # Plot Dataset 2 patients
#     plt.subplot(1, 2, 2)
#     plt.scatter(
#         reduced_data_arr2[:, 0],
#         reduced_data_arr2[:, 1],
#         c='grey',
#         marker='o',
#         alpha=0.5,
#         edgecolors='k',
#         s=100,
#         label='Mainz Data'
#     )
#     plt.scatter(
#         reduced_data_arr2[closest_indices, 0],
#         reduced_data_arr2[closest_indices, 1],
#         c='red',
#         marker='^',
#         alpha=0.7,
#         edgecolors='k',
#         s=100,
#         label='Closest to UMM Test Dataset'
#     )
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.title('Mainz Data')
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
#     plt.legend()
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.show()
#
#     for i in range(len(xs_pro)):
#         xs_pro[i] = xs_pro[i][indices.squeeze()]
#         ys_pro[i] = ys_pro[i][indices.squeeze()]
#
#     return xs_pro, ys_pro


def select_closest_patients_from_arrays(xs_test, xs_pro, ys_test, ys_pro):
    """
    Select the closest patients from xs_pro to those in xs_test based on their features
    and plot the clusters using binary labels.

    Parameters:
    xs_test (np.ndarray): Array containing data for dataset 1 (e.g., 52 patients).
    xs_pro (np.ndarray): Array containing data for dataset 2 (e.g., 284 patients).
    ys_test (np.ndarray): Array containing binary labels for dataset 1 (e.g., 52 patients).
    ys_pro (np.ndarray): Array containing binary labels for dataset 2 (e.g., 284 patients).

    Returns:
    list: List of indices from xs_pro that are closest to the patients in xs_test.
    """
    # Take first element of both xs_test and xs_pro
    arr1 = xs_test[0]
    arr2 = xs_pro[0]

    label1 = ys_test[0]
    label2 = ys_pro[0]

    # Combine the arrays for processing
    combined_arr = np.vstack((arr1, arr2))

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(combined_arr)

    # Perform PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(standardized_data)

    # Extract PCA components for Dataset 1 and Dataset 2
    reduced_data_arr1 = reduced_data[:len(arr1), :]
    reduced_data_arr2 = reduced_data[len(arr1):, :]

    """Use K-Nearest Neighbor approach"""
    arr1_scaled = scaler.transform(standardized_data[:len(arr1), :])
    arr2_scaled = scaler.transform(standardized_data[len(arr1):, :])

    # Use NearestNeighbors to find the closest patients
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(arr2_scaled)
    distances, indices = nbrs.kneighbors(arr1_scaled)

    # Determine the common range for x and y axes
    x_min = min(reduced_data[:, 0]) - 1
    x_max = max(reduced_data[:, 0]) + 1
    y_min = min(reduced_data[:, 1]) - 1
    y_max = max(reduced_data[:, 1]) + 1

    # Plot the PCA-reduced data with color coding for binary labels
    plt.figure(figsize=(12, 6))

    # Plot Dataset 1 patients with labels
    plt.subplot(1, 2, 1)
    plt.scatter(
        reduced_data_arr1[label1 == 0, 0],
        reduced_data_arr1[label1 == 0, 1],
        c='blue',
        marker='o',
        alpha=0.7,
        edgecolors='k',
        s=100,
        label='No Cirrhosis'
    )
    plt.scatter(
        reduced_data_arr1[label1 == 1, 0],
        reduced_data_arr1[label1 == 1, 1],
        c='red',
        marker='o',
        alpha=0.7,
        edgecolors='k',
        s=100,
        label='Cirrhosis'
    )
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('UMM Test Dataset')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)

    # Plot Dataset 2 patients with default grey color
    plt.subplot(1, 2, 2)
    plt.scatter(
        reduced_data_arr2[:, 0],
        reduced_data_arr2[:, 1],
        c='grey',
        marker='o',
        alpha=0.5,
        edgecolors='k',
        s=100,
        label='All Patients'
    )

    # Highlight closest patients with their respective labels
    closest_points = reduced_data_arr2[indices.squeeze()]
    closest_labels = label2[indices.squeeze()]

    plt.scatter(
        closest_points[closest_labels == 0, 0],
        closest_points[closest_labels == 0, 1],
        c='blue',
        marker='^',
        alpha=0.7,
        edgecolors='k',
        s=100,
        label='Clostest Patients - No Cirrhosis'
    )
    plt.scatter(
        closest_points[closest_labels == 1, 0],
        closest_points[closest_labels == 1, 1],
        c='red',
        marker='^',
        alpha=0.7,
        edgecolors='k',
        s=100,
        label='Clostest Patients - Cirrhosis'
    )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Mainz Data with Highlighted Closest Patients')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    for i in range(len(xs_pro)):
        xs_pro[i] = xs_pro[i][indices.squeeze()]
        ys_pro[i] = ys_pro[i][indices.squeeze()]

    return xs_pro, ys_pro
