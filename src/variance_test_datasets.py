import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.multivariate.manova import MANOVA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your datasets
# Replace 'path_to_dataset1.csv' and 'path_to_dataset2.csv' with your actual file paths
#
df1 = pd.read_csv('../data/preprocessed_mice_fib_test/test_cirrhosis_0.csv')  # 52 patients
# df1_1 = pd.read_csv('../data/preprocessed_mice_fib_train/train_cirrhosis_0.csv')  # 52 patients
# df1_2 = pd.read_csv('../data/preprocessed_mice_fib_val/val_cirrhosis_0.csv')  # 52 patients
# df1 = pd.concat([df1, df1_1, df1_2], ignore_index=True)
df2 = pd.read_csv('../data/preprocessed_mice_fib_prospective/prospective_cirrhosis_0.csv')  # 284 patients
#
# # Rename columns to remove special characters (e.g., %, spaces, parentheses)
# df1.columns = df1.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(' ', '')
# df2.columns = df2.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(' ', '')
#
# # Ensure df1 and df2 have the same columns (if not, rename/match columns accordingly)
# assert list(df1.columns) == list(df2.columns), "Datasets do not have matching columns"
#
# # Statistical Comparison of Each Marker Between the Two Groups
# print("Performing t-tests (or Mann-Whitney U tests if data is non-normal)...")
# p_values_ttest = {}
# p_values_mannwhitney = {}
#
# for marker in df1.columns:
#     stat_ttest, p_value_ttest = ttest_ind(df1[marker], df2[marker])
#     p_values_ttest[marker] = p_value_ttest
#
#     stat_mannwhitney, p_value_mannwhitney = mannwhitneyu(df1[marker], df2[marker])
#     p_values_mannwhitney[marker] = p_value_mannwhitney
#
# print("T-test p-values:")
# print(p_values_ttest)
#
# print("\nMann-Whitney U test p-values:")
# print(p_values_mannwhitney)
#
# # Multivariate Analysis of Variance (MANOVA)
# print("\nPerforming MANOVA...")
# # Add a group column for identification
# df1['group'] = 1
# df2['group'] = 2
# combined_df = pd.concat([df1, df2])
#
# # Create a formula for MANOVA using all marker columns
# formula = ' + '.join(df1.columns[:-1]) + ' ~ group'
# manova = MANOVA.from_formula(formula, data=combined_df)
# print(manova.mv_test())
#
# # Clustering the Combined Data
# print("\nStandardizing and performing clustering...")
# # Remove the 'group' column for clustering
# combined_data = combined_df.drop(columns=['group'])
#
# # Standardize the data
# scaler = StandardScaler()
# standardized_data = scaler.fit_transform(combined_data)
#
# # Determine the optimal number of clusters using the Elbow Method
# inertia = []
# for n_clusters in range(1, 10):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#     kmeans.fit(standardized_data)
#     inertia.append(kmeans.inertia_)
#
# # Plot the Elbow graph to determine the optimal number of clusters
# plt.plot(range(1, 10), inertia, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal Clusters')
# plt.show()
#
# # Let's assume optimal_clusters based on elbow plot is 3
# optimal_clusters = 3  # Update this based on the elbow plot result
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
# kmeans.fit(standardized_data)
# labels = kmeans.labels_
#
# # Adding cluster labels back to the original data
# combined_df['cluster'] = labels
#
# # Visualizing clusters using PCA (for 2D visualization)
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(standardized_data)
#
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('Cluster Visualization with PCA')
# plt.show()
#
# # Analyzing the clustering results
# print("\nCluster distribution in each dataset:")
# df1['cluster'] = labels[:len(df1)]
# df2['cluster'] = labels[len(df1):]
#
# print("Cluster distribution in Dataset 1 (52 patients):")
# print(df1['cluster'].value_counts())
#
# print("\nCluster distribution in Dataset 2 (284 patients):")
# print(df2['cluster'].value_counts())


# Rename columns to remove special characters and spaces
df1.columns = df1.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(' ', '')
df2.columns = df2.columns.str.replace(r'[^\w\s]', '_', regex=True).str.replace(' ', '')

# Add 'source' column to identify which dataset the rows belong to
df1['source'] = 'Dataset1'
df2['source'] = 'Dataset2'

# Combine the datasets for clustering
combined_df = pd.concat([df1, df2], ignore_index=True)

# Standardize the data (excluding the 'source' column)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(combined_df.drop(columns=['source']))

# Perform KMeans clustering
optimal_clusters = 3  # As determined from the Elbow method or other analysis
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
kmeans.fit(standardized_data)

# Add cluster labels back to the combined dataframe
combined_df['cluster'] = kmeans.labels_

# Separate combined data back into df1 and df2 for analysis
df1_clusters = combined_df[combined_df['source'] == 'Dataset1']['cluster']
df2_clusters = combined_df[combined_df['source'] == 'Dataset2']['cluster']

# Display the distribution of clusters in each dataset
print("Cluster distribution in Dataset 1:")
print(df1_clusters.value_counts())

print("\nCluster distribution in Dataset 2:")
print(df2_clusters.value_counts())

# Visualize clusters using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(standardized_data)

# Determine shared y-axis limits based on PCA data range
y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()

# Create a figure with two vertically stacked subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)

# Plot Dataset 1
scatter1 = ax1.scatter(
    reduced_data[combined_df['source'] == 'Dataset1', 0],
    reduced_data[combined_df['source'] == 'Dataset1', 1],
    c=combined_df[combined_df['source'] == 'Dataset1']['cluster'],
    cmap='cool',  # Cool colormap for Dataset 1
    marker='o',   # Circle marker for Dataset 1
    alpha=0.7,    # Set transparency
    edgecolors='k',  # Black edge color
    s=100,        # Marker size
)
ax1.set_title('Cluster Visualization for UMM Dataset')
ax1.set_ylabel('PCA Component 2')
ax1.grid(True)
ax1.set_ylim(y_min - 1, y_max + 1)  # Set the same y-axis range

# Plot Dataset 2
scatter2 = ax2.scatter(
    reduced_data[combined_df['source'] == 'Dataset2', 0],
    reduced_data[combined_df['source'] == 'Dataset2', 1],
    c=combined_df[combined_df['source'] == 'Dataset2']['cluster'],
    cmap='autumn',  # Autumn colormap for Dataset 2
    marker='^',     # Triangle marker for Dataset 2
    alpha=0.7,      # Set transparency
    edgecolors='k',  # Black edge color
    s=100,          # Marker size
)
ax2.set_title('Cluster Visualization for Mainz Dataset')
ax2.set_xlabel('PCA Component 1')
ax2.set_ylabel('PCA Component 2')
ax2.grid(True)
ax2.set_ylim(y_min - 1, y_max + 1)  # Set the same y-axis range

# Adjust layout and show plot
plt.tight_layout()
plt.show()