from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from src.select_test_datasets import *
from sklearn.model_selection import StratifiedKFold
from src.utils.plots import *
from src.utils.smote import apply_smote_to_datasets

pd.options.mode.chained_assignment = None


def create_scores(df_pro):
    mapping = {
        'F0': 0,
        # 'F0-1': 0
        'F1': 1,
        # 'F1-F2': 1,
        'F2': 2,
        # 'F2-F3': 2,
        'F3': 3,
        # 'F3-F4': 3,
        'F4': 4
    }

    # Remove rows where 'Micro' contains a hyphen
    df_pro = df_pro[~df_pro['Micro'].str.contains('-')]

    # Map the 'Micro' column to integers using the predefined mapping
    df_pro['Micro'] = df_pro['Micro'].map(mapping)

    return df_pro


def preprare_data(classification_type, shap_selected, scaling, finetune=False, select_patients=False, smote=False):
    # Load data
    if finetune:
        df = pd.read_excel('../data/20240813-FibrosisDB(302_Patients).xlsx')
        df = rename_column_names(df)
        df = create_scores(df)
        df_umm_1 = pd.read_excel('../data/20231129 Lap und Histo Daten von Ines Tuschner.xlsx')
        df_umm_2 = pd.read_excel('../data/202403 Lap und Histo Daten von Ines Tuschner.xlsx')
        df_umm_2 = df_umm_2[['HbA1c (%)', 'Glucose in plasma (mg/dL)', 'LDL- Cholesterin (mg/dL)']]
        df_umm = pd.concat([df_umm_1, df_umm_2], axis=1)

    else:
        df = pd.read_excel('../data/20231129 Lap und Histo Daten von Ines Tuschner.xlsx')
        df2 = pd.read_excel('../data/202403 Lap und Histo Daten von Ines Tuschner.xlsx')
        df_pro = pd.read_excel('../data/20240813-FibrosisDB(302_Patients).xlsx')
        df_pro = rename_column_names(df_pro)
        df_pro = create_scores(df_pro)

        df2 = df2[['HbA1c (%)', 'Glucose in plasma (mg/dL)', 'LDL- Cholesterin (mg/dL)']]
        df = pd.concat([df, df2], axis=1)

    print(f'\n----- length of original dataframe: {len(df)} -----\n')

    if finetune:
        # Split df into train_val_df and test_df
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        train_df_umm, test_df_umm = train_test_split(df_umm, test_size=0.1, random_state=42)

    else:
        # Split df into train_val_df and test_df
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Split train_val_df into train_df and val_df
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

    if finetune:
        data_splits = ['train_ft', 'val_ft', 'test_ft']
    else:
        data_splits = ['train', 'val', 'test']

    for split in data_splits:
        if not os.path.isdir(f'../data/preprocessed_no_mice_{split}/'):
            os.mkdir(f'../data/preprocessed_no_mice_{split}/')
        if not os.path.isdir(f'../data/preprocessed_mice_fib_{split}/'):
            os.mkdir(f'../data/preprocessed_mice_fib_{split}/')

    if not os.path.isdir(f'../data/preprocessed_no_mice_prospective/'):
        os.mkdir(f'../data/preprocessed_no_mice_prospective/')
    if not os.path.isdir(f'../data/preprocessed_mice_fib_prospective/'):
        os.mkdir(f'../data/preprocessed_mice_fib_prospective/')

    if finetune:
        xs_train, ys_train, df_cols, scaler = preprocess(train_df, 'train_ft', classification_type=classification_type,
                                                         scaling=scaling, scaler=None, shap_selected=shap_selected)
        xs_val, ys_val, df_cols, scaler = preprocess(val_df, 'val_ft', classification_type=classification_type,
                                                     scaling=scaling, scaler=scaler, shap_selected=shap_selected)
        xs_test, ys_test, df_cols, scaler = preprocess(test_df, 'test_ft', classification_type=classification_type,
                                                       scaling=scaling, scaler=scaler, shap_selected=shap_selected)

        xs_test_umm, ys_test_umm, df_cols_umm, scaler = preprocess(test_df_umm, 'test',
                                                                   classification_type=classification_type,
                                                                   scaling=scaling, scaler=scaler,
                                                                   shap_selected=shap_selected)

    else:
        xs_train, ys_train, df_cols, scaler = preprocess(train_df, 'train', classification_type=classification_type,
                                                         scaling=scaling, scaler=None, shap_selected=shap_selected)
        xs_val, ys_val, df_cols, scaler = preprocess(val_df, 'val', classification_type=classification_type,
                                                     scaling=scaling, scaler=scaler, shap_selected=shap_selected)
        xs_test, ys_test, df_cols, scaler = preprocess(test_df, 'test', classification_type=classification_type,
                                                       scaling=scaling, scaler=scaler, shap_selected=shap_selected)

        xs_pro, ys_pro, df_cols, scaler = preprocess(df_pro, 'prospective', classification_type=classification_type,
                                                     scaling=scaling, scaler=scaler, shap_selected=shap_selected,
                                                    select_closest_patients_from_mainz=select_patients, smote=smote)

    # concat all data files to one processed_data csv
    merged_no_mice_df = pd.concat([pd.read_csv(f'../data/preprocessed_no_mice_train/train_{classification_type}.csv'),
                                   pd.read_csv(f'../data/preprocessed_no_mice_val/val_{classification_type}.csv'),
                                   pd.read_csv(f'../data/preprocessed_no_mice_test/test_{classification_type}.csv'),
                                   pd.read_csv(
                                       f'../data/preprocessed_no_mice_prospective/prospective_{classification_type}.csv')],
                                  ignore_index=True)
    merged_no_mice_df.to_csv('../data/preprocessed_no_mice_data.csv', index=False)

    merged_mice_fib_df = pd.concat(
        [pd.read_csv(f'../data/preprocessed_mice_fib_train/train_{classification_type}_0.csv'),
         pd.read_csv(f'../data/preprocessed_mice_fib_val/val_{classification_type}_0.csv'),
         pd.read_csv(f'../data/preprocessed_mice_fib_test/test_{classification_type}_0.csv'),
         pd.read_csv(f'../data/preprocessed_mice_fib_prospective/prospective_{classification_type}_0.csv')],
        ignore_index=True)
    merged_mice_fib_df.to_csv('../data/preprocessed_mice_fib_data.csv', index=False)

    # Check FIB4 Stage population regarding fibrosis or cirrhosis
    counts = merged_mice_fib_df['Fib4 Stages'].value_counts()

    # Get the total number of entries in the column
    total = len(merged_mice_fib_df['Fib4 Stages'])

    # Calculate the percentages
    if classification_type == 'three_stage':
        percentage_2 = (counts.get(2, 0) / total) * 100
    percentage_1 = (counts.get(1, 0) / total) * 100
    percentage_0 = (counts.get(0, 0) / total) * 100

    print('-------------------------------------------')
    if classification_type == 'three_stage':
        print(f"Population of {classification_type} according to FIB4: {percentage_2:.2f}%")
    print(f"Population of {classification_type} according to FIB4: {percentage_1:.2f}%")
    print(f"Population of {classification_type} according to FIB4: {percentage_0:.2f}%\n")

    # Check Micro Stage population regarding fibrosis or cirrhosis
    counts = merged_mice_fib_df['Micro'].value_counts()

    # Get the total number of entries in the column
    total = len(merged_mice_fib_df['Micro'])

    # Calculate the percentages
    if classification_type == 'three_stage':
        percentage_2 = (counts.get(2, 0) / total) * 100
    percentage_1 = (counts.get(1, 0) / total) * 100
    percentage_0 = (counts.get(0, 0) / total) * 100

    print('-------------------------------------------')
    if classification_type == 'three_stage':
        print(f"Population of {classification_type} according to Micro: {percentage_2:.2f}%")
    print(f"Population of {classification_type} according to Micro: {percentage_1:.2f}%")
    print(f"Population of {classification_type} according to Micro: {percentage_0:.2f}%\n")
    print('-------------------------------------------')

    if not finetune and select_patients:
        xs_pro, ys_pro = select_closest_patients_from_arrays(xs_test, xs_pro, ys_test, ys_pro)
    if not finetune:
        return xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols
    else:
        return xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, xs_test_umm, ys_test_umm, df_cols


def preprocess(df, data_type='train', classification_type='fibrosis', scaling=False, scaler=None,
               shap_selected=False, select_closest_patients_from_mainz=False, smote=False):
    """
    Main function for preprocessing. Preprocesses the data.
    Args:
        df (pd.DataFrame): data to preprocess.
        data_type (str): 'train', 'val', 'test', 'prospective'
        classification_type (str): .
        scaling (bool): whether to use scaling
        scaler: standard scaler to use for scaling imputed datasets
        shap_selected (bool): whether to select top-n shap biomarkers for testing validation decrease compared
        to all biomarkers as input
        select_closest_patients_from_mainz: whether to select closest patients
        smote: if smote is used for minority upsampling

    Returns:
        xs (list): m dataframes.
        ys (list): m labels.
        cols (list): column names.
    """

    print(f'\n ----- processing {data_type} data ----- \n')
    df = df.reset_index()

    # Convert age from birth data
    df = calculate_age(df)

    # Clean df from wrong entries (>, <, 2 x ., etc)
    df, (indices, values, operator_cols) = clean_df(df)
    df = df.astype(float)

    # Remove NANs from Micro column
    len_before = len(df)
    df = df.dropna(subset='Micro')
    len_after = len(df)

    print(f'NAN Micro: Dropped {len_before - len_after} number of rows {len_before}, {len_after}')

    len_before = len(df)
    df = drop_rows_with_high_missing_data(df)
    len_after = len(df)
    print(f'Missing Entries Threshold: Dropped {len_before - len_after} number of rows {len_before}, {len_after}')

    df = df.astype(float)
    df['Micro'] = df['Micro'].astype(int, errors='ignore')

    # # Temporary subsample F4 patients
    # if data_type == 'prospective':
    #     # Plotting the histogram
    #     plt.figure(figsize=(8, 6))
    #     bins = range(df['Micro'].min(), df['Micro'].max())
    #
    #     n, bins, patches = plt.hist(df['Micro'], bins=bins, edgecolor='black')
    #
    #     # Set the x-ticks to be in the middle of each bin
    #     plt.xticks([bin for bin in bins[:-1]])
    #     plt.title('Histogram of Liver Stages - Data Mainz')
    #     plt.xlabel('Liver Stage')
    #     plt.ylabel('Frequency')
    #     plt.show()
    #
    #     breakpoint()
    #
    #     # Step 1: Filter the dataframe to get rows where 'Micro' is 4
    #     f4_patients = df[df['Micro'] == 4]
    #
    #     # Step 2: Take the first 60 occurrences
    #     f4_patients_first_60 = f4_patients.head(60)
    #
    #     # Step 3: Drop all rows with 'Micro' == 4 from the original dataframe
    #     df = df[df['Micro'] != 4]
    #
    #     # Step 4: Concatenate the first 60 occurrences back into the dataframe
    #     df = pd.concat([df, f4_patients_first_60])
    #
    #     # Optionally reset the index
    #     df.reset_index(drop=True, inplace=True)

    # Binarize Micro
    df['Micro'] = df['Micro'].apply(lambda x: categorize_micro(x, classification_type=classification_type))

    if not os.path.isdir(f'../data/preprocessed_mice_fib_{data_type}'):
        os.mkdir(f'../data/preprocessed_mice_fib_{data_type}')

    # Extract preprocessed
    df.to_csv(f'../data/preprocessed_no_mice_{data_type}/{data_type}_{classification_type}.csv', index=False)

    dfs = mice(df, 10)

    # TODO: refactor preprocessing and smote implementation
    if data_type == 'prospective' and smote:
        dfs_, x_list, y_list = [], [], []
        for idx, df in enumerate(dfs):
            x = df.drop(columns=['Micro']).to_numpy()
            y = df['Micro'].to_numpy()
            x_list.append(x)
            y_list.append(y.astype(int))

        # Apply SMOTE
        x_list, y_list = apply_smote_to_datasets(x_list, y_list)

        for idx, df in enumerate(dfs):
            # Convert back to DataFrame
            df_smote = pd.DataFrame(x_list[idx], columns=df.drop(
                columns=['Micro']).columns)  # Set original column names for features
            df_smote['Micro'] = y_list[idx]  # Add the target column back
            dfs_.append(df_smote)
        dfs = dfs_

    ys = []
    xs = []

    if scaling and scaler is None and data_type == 'train':
        scaler = StandardScaler()

    for idx, df in enumerate(dfs):
        y = df['Micro']
        y = y.astype(int)
        x = df.drop('Micro', axis=1)

        if shap_selected:
            # Columns to select
            columns_to_select = ['Thrombozyten (Mrd/l)', 'MCV (fl)', 'INR']
            # columns_to_select = ['ASAT (U/I)', 'ALAT (U/I)', 'Age', 'Thrombozyten (Mrd/l)']  # FIB4 Marker
            # New DataFrame with selected columns
            x = x[columns_to_select]

        if scaling:
            # Fit the scaler on the training data and transform
            if data_type == 'train':
                x_scaled = scaler.fit_transform(x)
            else:
                x_scaled = scaler.transform(x)
            xs.append(pd.DataFrame(x_scaled, columns=x.columns))

        else:
            xs.append(x)

        ys.append(y)

        df = calculate_fib4(df, classification_type)
        df.to_csv(f'../data/preprocessed_mice_fib_{data_type}/{data_type}_{classification_type}_{idx}.csv', index=False)

    if any(substring in data_type for substring in ['test', 'prospective']):
        analyze_fib4(dfs, classification_type=classification_type, data_type=data_type,
                     select_patients_from_mainz=select_closest_patients_from_mainz, smote=smote)

    cols = [*xs[0].columns]

    xs = [df.values for df in xs]
    ys = [df.values for df in ys]

    print(f'Length of data set {len(ys[0])}')

    return xs, ys, cols, scaler


def analyze_fib4(dfs, classification_type='fibrosis', data_type='train', select_patients_from_mainz=False,
                 subgroup_analysis=True, smote=False):
    """
    Computes the FIB4 classification performances on the data sets.
    Args:
        dfs (pandas.Dataframe): list of dataframes.

    Returns:
        None
    """
    if select_patients_from_mainz and data_type == 'prospective':
        df_test_0 = pd.read_csv(f'../data/preprocessed_mice_fib_test/test_{classification_type}_0.csv')
        df_pro_0 = dfs[0]

        # Select closest patients
        df_pro_0 = select_closest_patients(df_test_0, df_pro_0)

        # # Select oldest patients
        # df_pro_0 = select_oldest_patients(df_test_0, df_pro_0)

        # Select highest platelet patients
        # df_pro_0 = select_highest_platelet_patients(df_test_0, df_pro_0)

        dfs = [df_pro_0] * 10

    if not os.path.exists(f'outputs/fib4'):
        os.makedirs(f'outputs/fib4')

    # for i, df in enumerate(dfs):
    # Take first imputed df
    df = dfs[0]

    """ Subgroup Analysis """
    if data_type == 'prospective' and subgroup_analysis:
        df = categorize_patients(df)
        df_test_0 = pd.read_csv(f'../data/preprocessed_mice_fib_test/test_{classification_type}_0.csv')
        df_test = categorize_patients(df_test_0)

        # Separate the biomarkers into the respective groups
        groups_df = df.groupby('Classification')[['Thrombozyten (Mrd/l)', 'ASAT (U/I)', 'ALAT (U/I)', 'Age']]
        groups_df_test = df_test.groupby('Classification')[['Thrombozyten (Mrd/l)', 'ASAT (U/I)', 'ALAT (U/I)', 'Age']]

        # Calculate summary statistics for each group
        summary_stats_df = groups_df.describe()
        summary_stats_df_test = groups_df_test.describe()

        print(summary_stats_df)
        print(summary_stats_df_test)

        # Define a consistent order for the classifications
        classification_order = ['FP', 'TP', 'TN', 'FN']

        # Visualize the distributions using box plots
        for marker in ['Thrombozyten (Mrd/l)', 'ASAT (U/I)', 'ALAT (U/I)', 'Age']:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)  # Create side-by-side subplots

            # Plot for the first dataframe with consistent order
            sns.boxplot(x='Classification', y=marker, data=df, ax=axes[0], order=classification_order)
            axes[0].set_title(f'{marker} Distribution (Mainz)')
            axes[0].set_xlabel('Classification')
            axes[0].set_ylabel(marker)

            # Plot for the second dataframe with consistent order
            sns.boxplot(x='Classification', y=marker, data=df_test, ax=axes[1], order=classification_order)
            axes[1].set_title(f'{marker} Distribution (UMM Test)')
            axes[1].set_xlabel('Classification')
            axes[1].set_ylabel(marker)

            # Apply shading consistently
            for ax in axes:
                # Add green shading for TP and TN, red for FP and FN based on their dynamic positions
                ax.axvspan(classification_order.index('TP') - 0.5, classification_order.index('TP') + 0.5,
                           color='green', alpha=0.2)
                ax.axvspan(classification_order.index('TN') - 0.5, classification_order.index('TN') + 0.5,
                           color='green', alpha=0.2)
                ax.axvspan(classification_order.index('FP') - 0.5, classification_order.index('FP') + 0.5,
                           color='red', alpha=0.2)
                ax.axvspan(classification_order.index('FN') - 0.5, classification_order.index('FN') + 0.5,
                           color='red', alpha=0.2)

            plt.suptitle(f'{marker} Distribution by Classification Group for FIB4 CM')
            plt.tight_layout()
            plt.show()

    skf = StratifiedKFold(n_splits=5)

    # To store metrics
    cms = []
    reports = []

    # Extract the actual and predicted labels
    y_true = df['Micro']
    y_pred = df['Fib4 Stages']

    # Perform cross-validation
    # Use for stratified_split the train_index (length is 80% of all samples) to calculate the CMs
    for split_index, _ in skf.split(np.zeros(len(y_true)), y_true):
        y_true_split = y_true.iloc[split_index]
        y_pred_split = y_pred.iloc[split_index]

        # Compute confusion matrix and classification report
        cm = confusion_matrix(y_true_split, y_pred_split)
        report = classification_report(y_true_split, y_pred_split, output_dict=True)

        # Append metrics
        cms.append(cm)
        reports.append(report)

    plot_cm(cms[0], 'fib4', classification_type=classification_type)

    with open(f'outputs/fib4/fib4_{classification_type}.csv', 'w') as f:
        for (cm, report) in zip(cms, reports):
            f.write(str(report))
            f.write(str(cm))

    acc, f1, ppv, tpr = [], [], [], []
    for report in reports:
        acc.append(report['accuracy'])
        f1.append(report['macro avg']['f1-score'])
        ppv.append(report['macro avg']['precision'])
        tpr.append(report['macro avg']['recall'])

    performance_string = f'\nFIB4\n' \
                         f'Average ACC: {np.round(np.average(acc) * 100, 2)} | Std ACC: {np.round(np.std(acc) * 100, 2)},\n' \
                         f'Average F1: {np.round(np.average(f1) * 100, 2)} | Std F1: {np.round(np.std(f1) * 100, 2)},\n' \
                         f'Average PPV: {np.round(np.average(ppv) * 100, 2)} | Std PPV: {np.round(np.std(ppv) * 100, 2)},\n' \
                         f'Average TPR: {np.round(np.average(tpr) * 100, 2)} | Std TPR: {np.round(np.std(tpr) * 100, 2)}\n'
    print(performance_string)

    with open(f'outputs/fib4/fib4_{classification_type}_performance_metrics.txt', 'w') as f:
        f.write(performance_string)
        f.close()


def categorize_patients(df):
    """
    Categorize patients into FN, FP, TP, TN based on the FIB4 prediction and true stage.
    """
    conditions = [
        (df['Micro'] == 1) & (df['Fib4 Stages'] == 0),  # False Negative
        (df['Micro'] == 0) & (df['Fib4 Stages'] == 1),  # False Positive
        (df['Micro'] == 1) & (df['Fib4 Stages'] == 1),  # True Positive
        (df['Micro'] == 0) & (df['Fib4 Stages'] == 0)  # True Negative
    ]
    choices = ['FN', 'FP', 'TP', 'TN']

    df['Classification'] = np.select(conditions, choices, default='Unclassified')
    return df


def categorize_micro(value, classification_type='fibrosis'):
    """
    Recategorizes the binary classification task to binary.
    Args:
        value (int): Single entry of dataframe.
        classification_type (str): either "fibrosis" or "cirrhosis"

    Returns:
        int or np.nan: Binary return.
    """
    if classification_type == 'cirrhosis':
        if 0 <= value <= 3:
            return 0
        elif value > 3:
            return 1
    elif classification_type == 'fibrosis':
        if 0 <= value <= 1:
            return 0
        elif value > 1:
            return 1
    elif classification_type == 'two_stage':
        if 0 <= value <= 2:
            return 0
        elif value > 2:
            return 1
    elif classification_type == 'three_stage':
        if 0 <= value <= 1:
            return 0
        elif 2 <= value <= 3:
            return 1
        elif value > 3:
            return 2
    else:
        raise ValueError(f'classification_type {classification_type} is not implemented!')


def calculate_fib4(df, classification_type='fibrosis'):
    """
    Calculate the FIB4 Score accordingly.
    Args:
        df (pd.DataFrame): Data.
        classification_type (str): 'fibrosis' or 'cirrhosis'

    Returns:
        df (pd.DataFrame): Previous df including a new column with the according FIB4 Stages.
    """
    df['Fib4'] = (df['Age'] * df['ASAT (U/I)']) / \
                 (df['Thrombozyten (Mrd/l)'] * np.sqrt(df['ALAT (U/I)']))
    df['Fib4 Stages'] = df['Fib4'].apply(lambda x: calculate_fib_stages(x, classification_type=classification_type))

    return df


def calculate_fib_stages(value, classification_type='fibrosis'):
    """
    Calculates the FIB4 classification value based on the FIB4 Score.
    Args:
        value (float): entry in DataFrame.
        classification_type (str): either "fibrosis" or "cirrhosis"

    Returns:
        0 or 1 (int): Classification for this specific entry.
    """
    if classification_type == 'cirrhosis':
        if value < 3.25:
            return 0
        elif value >= 3.25:
            return 1
    elif classification_type == 'fibrosis':
        if value < 1.45:
            return 0
        elif value >= 1.45:
            return 1
    elif classification_type == 'two_stage':
        if value < 2.67:
            return 0
        elif value >= 2.67:
            return 1
    elif classification_type == 'three_stage':
        if value < 1.45:
            return 0
        elif 1.45 <= value <= 3.25:
            return 1
        elif value >= 3.25:
            return 2
    else:
        raise ValueError(f'classification_type {classification_type} is not implemented!')


def calculate_age(df):
    """
    Calculates the age of all the patients.
    Args:
        df (pd.DataFrame): Data.

    Returns:
        df (pd.DataFrame): Previous data including a new column "Age".
    """
    # Prospective study data set already has "Age" column
    if 'Age' not in df.columns:
        df["Geb.Datum"] = pd.to_datetime(df["Geb.Datum"])
        current_year = 2023
        df["Age"] = current_year - df["Geb.Datum"].dt.year
        df = df.drop('Geb.Datum', axis=1)
        df = df.drop('Blutentnahme', axis=1)

    return df


def keep_samples_three_months(df):
    """
    Filter DataFrame to keep only entries where samples were taken within the last three months
    based on the 'Blutentnahme' column.

    Args:
        df (pd.DataFrame): Input DataFrame containing columns with dates.

    Returns:
        DataFrame: DataFrame with filtered entries, retaining only those where samples were taken within the last three months from the date specified in the 'Blutentnahme' column.
    """
    # Extract dates from other columns (assuming consistent format)
    for col in df.filter(like='col'):
        df[col + '_date'] = df[col].apply(extract_date)

    # Convert 'blutentnahme' to datetime format
    df.loc[:, 'blutentnahme_datetime'] = pd.to_datetime(
        df['Blutentnahme'], errors='coerce')

    # Select the date part only (without time)
    df.loc[:, 'blutentnahme_date'] = df['blutentnahme_datetime'].dt.date

    # Calculate the difference between dates
    for col in df.filter(like='col_date'):
        df['diff_' +
           col.split('_')[0]] = pd.to_timedelta(df['blutentnahme_date'] -
                                                df[col])

    # Filter for entries within 3 months and update original columns
    three_months = pd.Timedelta(days=3 * 30)
    for col in df.filter(like='col'):
        df[col] = np.where(df['diff_' + col.split('_')[0]]
                           <= three_months, df[col], np.nan)

    # Drop unnecessary columns (optional)
    df = df.drop(columns=['blutentnahme_datetime',
                          *(col for col in df.filter(like='col_date'))])
    df = df.drop(['Blutentnahme', 'blutentnahme_date'], axis=1)
    return df


def extract_date(text):
    """
    Extract a date from a string enclosed in parentheses and convert it to a datetime object.

    Args:
        text (str): Input text possibly containing a date enclosed in parentheses.

    Returns:
        datetime: Datetime object representing the extracted date if found; otherwise, 'NaT'.
    """
    if text and '(' in text:
        return pd.to_datetime(text[1:-1])
    else:
        return np.datetime64('NaT')


def clean_df(df):
    """
    Preprocesses the input DataFrame by performing several data cleaning and manipulation steps.

    Args:
        df (DataFrame): Input DataFrame containing the data to be preprocessed.

    Returns:
        DataFrame: Preprocessed DataFrame with cleaned and transformed data.
        tuple: Tuple containing indices and values after applying certain operations.
    """
    # Cols in order of SHAP importance
    cols = [
        'Thrombozyten (Mrd/l)',
        'MCV (fl)',
        'Quick (%)',
        'INR',
        #        'Glucose in plasma (mg/dL)',
        'Leukozyten (Mrd/l)',
        'ASAT (U/I)',
        'PTT (sek)',
        'ALAT (U/I)',
        'Age',
        'IgG (g/l)',
        'Albumin (g/l)',
        'HbA1c (%)',
        'Bilrubin gesamt (mg/dl)',
        'AP (U/I)',
        'Harnstoff',
        'Hb (g/dl)',
        'Kalium',
        'GGT (U/I)',
        'Kreatinin (mg/dl)',
        'GRF (berechnet) (ml/min)',

        'Micro',
    ]

    indices_list, values_list = [], []

    df = df[cols]

    # Function does not do anything, but removes a datetime column so astype(float) works
    # df = keep_samples_three_months(df)

    # Deletes everything after there is a parantheses
    df = df.map(lambda x: str(x).split('(')[0].strip())

    df.loc[:, 'Micro'] = pd.to_numeric(df['Micro'], errors='coerce')

    operator_cols = ['GRF (berechnet) (ml/min)', 'ALAT (U/I)', 'Quick (%)', 'HbA1c (%)']

    df = df.astype(str)

    for operator_col in operator_cols:
        df, indices, values = handle_operator(df, operator_col)

        indices_list.append(indices)
        values_list.append(values)

    def delete_after_whitespace(entry):
        return entry.split(maxsplit=1)[0]

    df = df.astype(str)

    # After whitespace there are dates in the data set
    df = df.map(delete_after_whitespace)

    # Some decimal values use commas instead of points
    df = df.replace(',', '.', regex=True)
    # Specific values that need to be replaced
    df = df.replace('kein', np.nan)
    # 32.4 would make sense acccording to other values
    df = df.replace('32.420.2.)', 32.4)
    # There are some 'neg' values for PTT
    df = df.replace('neg', np.nan)

    return df, (indices_list, values_list, operator_cols)


def drop_rows_with_high_missing_data(df, missing_data_threshold=0.70):
    """
    Drops rows from a pandas DataFrame where the percentage of missing data exceeds a threshold.
    Used because there are a certain number of patients that have missing LAP completely.
    Some patients are also double or triple in the data set. This function deletes them too.

    Args:
        df (pandas.DataFrame): The DataFrame to clean.
        missing_data_threshold (float, optional): The percentage of missing data above which a row is dropped. Defaults to 0.7 (70%).

    Returns:
        pandas.DataFrame: The cleaned DataFrame with rows exceeding the missing data threshold removed.
    """

    # Drop rows where more "than missing_data_threshold" of data is missing
    df_cleaned = df.dropna(thresh=(1 - missing_data_threshold) * df.shape[1])

    return df_cleaned


def handle_operator(df, col):
    """
    Handle operator symbols (<, >) present in the specified column of the DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing the data to be processed.
        col (str): Name of the column where operator symbols need to be handled.

    Returns:
        DataFrame: DataFrame with operator symbols replaced with NaN in the specified column.
        array-like: Indices of rows where operator symbols were found.
        DataFrame: Subset of DataFrame containing rows with operator symbols in the specified column.
    """
    indices = df[df[col].str.contains('<|>', case=False)].index
    values = df[df[col].str.contains('<|>', case=False)]

    # Replace entries with "<" or ">" with NaN
    df.loc[df[col].str.contains('<|>', case=False), col] = np.nan

    return df, indices, values


def mice(data, m) -> pd.DataFrame:
    """
    Multiple Imputation by Chained Equations (MICE) algorithm for imputing missing values in the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame containing missing values.
        m (int): Number of imputations to perform.

    Returns:
        list: List of DataFrames containing imputed values for each iteration of the MICE algorithm.
    """
    data = data.replace([np.inf, -np.inf], np.nan)
    imp_dfs = []
    for i in range(m):
        imp = IterativeImputer(
            missing_values=np.nan,
            random_state=i,
            min_value=0,
            sample_posterior=True)
        imp_dfs.append(
            pd.DataFrame(
                imp.fit_transform(data),
                columns=data.columns))

    return imp_dfs


def rename_column_names(df_pro):
    cols_mapping = {
        'thrombozyten': 'Thrombozyten (Mrd/l)',
        'mcv': 'MCV (fl)',
        'quick': 'Quick (%)',
        'inr': 'INR',
        'leukozyten': 'Leukozyten (Mrd/l)',
        'asat': 'ASAT (U/I)',
        'ptt': 'PTT (sek)',
        'alat': 'ALAT (U/I)',
        'age': 'Age',
        'igg': 'IgG (g/l)',
        'albumin': 'Albumin (g/l)',
        'hba1c': 'HbA1c (%)',
        'bilirubin': 'Bilrubin gesamt (mg/dl)',
        'ap': 'AP (U/I)',
        'harnstoff': 'Harnstoff',
        'hb': 'Hb (g/dl)',
        'kalium': 'Kalium',
        'ggt': 'GGT (U/I)',
        'kreatinin': 'Kreatinin (mg/dl)',
        'gfr': 'GRF (berechnet) (ml/min)',

        'Fibrosen-grad': 'Micro'
    }

    df_pro.rename(columns=cols_mapping, inplace=True)

    return df_pro


if __name__ == '__main__':
    df = pd.read_excel(
        '../data/20231129 Lap und Histo Daten von Ines Tuschner.xlsx')
    df2 = pd.read_excel(
        '../data/202403 Lap und Histo Daten von Ines Tuschner.xlsx')
    df2 = df2[['HbA1c (%)', 'Glucose in plasma (mg/dL)',
               'LDL- Cholesterin (mg/dL)']]
    df = pd.concat([df, df2], axis=1)

    test_size = 0.15

    xs, ys, df_cols = preprocess(df)

    # Calculate the number of samples for the test set
    test_samples = int(len(xs[0]) * test_size)

    xs_train = []
    ys_train = []

    xs_test = []
    ys_test = []

    for x, y in zip(xs, ys):
        xs_train.append(x[:-test_samples])
        ys_train.append(y[:-test_samples])

        xs_test.append(x[-test_samples:])
        ys_test.append(y[-test_samples:])

    np.save('../data/xs_train', xs_train)
    np.save('../data/xs_test', xs_test)
    np.save('../data/ys_train', ys_train)
    np.save('../data/ys_test', ys_test)

    with open('../data/df_cols.pickle', 'wb') as f:
        pickle.dump(df_cols, f)
