import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
import itertools
import re
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score, f1_score
import ast

from utils.helper_functions import majority_vote
from preprocess import drop_rows_with_high_missing_data, keep_samples_three_months, calculate_age
from preprocess import clean_df, categorize_micro, mice, calculate_fib4


def print_missing_y(df):
    print('Dropping missing y values...')
    before_y_drop = len(df)
    df = df.dropna(subset='Micro')
    print(
        f'NAN y drop: {len(df)}, Rows dropped: {before_y_drop - len(df)}, Percentage {(before_y_drop - len(df)) / len(df)} ')


def print_high_missing_data(df):
    len_before = len(df)
    drop_rows_with_high_missing_data(df)
    len_after = len(df)
    print(
        f'High missing data: Dropped {len_before - len_after} number of rows. Before: {len_before}, After: {len_after}, Percentage: {1 - (len_after / len_before)}')


def print_keep_samples_3_months(df):
    len_before = len(df)
    keep_samples_three_months(df)
    len_after = len(df)
    print(
        f'Keep Samples three months:: Dropped {len_before - len_after} number of rows. Before: {len_before}, After: {len_after}, Percentage: {1 - (len_after / len_before)}')


def print_doubled_patients(df):
    len_before = len(df)
    df = df.drop_duplicates(subset=['Patientennummer'])
    len_after = len(df)
    print(
        f'Doubled patients: Dropped {len_before - len_after} number of rows. Before: {len_before}, After: {len_after}, Percentage: {1 - (len_after / len_before)}')


def filter_by_micro(df, y):
    if y not in [0, 1]:
        raise ValueError("y_value must be either 0 or 1")

    return df[df['Micro'] == y]


def imitate_preprocess(df, filter=None, classification_type='fibrosis'):
    df = df.reset_index()
    df = df.dropna(subset='Micro')
    df = calculate_age(df)

    df, (indices, values, operator_cols) = clean_df(df)
    df = df.astype(float)
    # df = df.dropna(subset='Micro')

    # df = drop_rows_with_high_missing_data(df)

    # df = df.astype(float)

    df['Micro'] = df['Micro'].astype(int, errors='ignore')
    df['Micro'] = df['Micro'].apply(lambda x: categorize_micro(x, classification_type=classification_type))

    if filter == 0:
        df = filter_by_micro(df, 0)
    elif filter == 1:
        df = filter_by_micro(df, 1)
    print('NUMBER OF SAMPLES: ', df['Micro'].value_counts(normalize=True), df['Micro'].value_counts())

    nan_mask = df.isna()
    true_means = df.mean()
    true_stds = df.std()
    true_df = df.copy()

    dfs = mice(df, 10)
    ys = []
    xs = []
    for df in dfs:
        y = df['Micro']
        y = y.astype(int)
        x = df.drop('Micro', axis=1)
        ys.append(y)
        xs.append(x)

    return true_df, dfs, nan_mask, true_means, true_stds, df.columns, (indices, values, operator_cols)


def get_mean_per_nan(imputed_values_with_positions, feature_name):
    means = []
    std = []
    for item in imputed_values_with_positions:
        if item['column'] == feature_name:
            means.append(np.mean(item['values']))
            std.append(np.std(item['values']))

    return means, std


def get_imputation_uncertainty(dfs, nan_mask, columns):
    imputed_values_with_positions = []

    for row in range(nan_mask.shape[0]):
        for col in range(nan_mask.shape[1]):
            if nan_mask.iat[row, col]:
                values_at_position = [df.iat[row, col] for df in dfs]
                imputed_values_with_positions.append({
                    'row': row,
                    'column': dfs[0].columns[col],
                    'values': values_at_position
                })

    means_std_per_feature = defaultdict()
    for feature_name in columns:
        individual_means, individual_stds = get_mean_per_nan(imputed_values_with_positions, feature_name)
        means_std_per_feature[feature_name] = (individual_means, individual_stds)

    return means_std_per_feature


def parse_condition(condition):
    match = re.match(r'([<>=!]=?|>=|<=)?\s*(-?\d+\.?\d*)', condition)
    if match:
        operator = match.group(1)
        number = float(match.group(2))
        return operator, number
    else:
        raise ValueError("Invalid condition format")


def get_percentage(lst, condition):
    operator, number = parse_condition(condition)

    operators = {
        '<': lambda x: x < number,
        '<=': lambda x: x <= number,
        '>': lambda x: x > number,
        '>=': lambda x: x >= number,
        '==': lambda x: x == number,
        '!=': lambda x: x != number
    }

    if operator not in operators:
        raise ValueError("Unsupported operator")

    filtered_list = [x for x in lst if operators[operator](x)]
    perc = len(filtered_list) / len(lst)

    return perc


def compare_operator_with_imputation(dfs, indices, values, operator_cols):
    operator_df = pd.concat(values)
    indices = list(itertools.chain.from_iterable(zip(indices, operator_cols)))

    # Extract the lists and corresponding strings
    lists = indices[0::2]  # Extracts lists at indices 0, 2, 4, 6
    strings = indices[1::2]  # Extracts strings at indices 1, 3, 5, 7

    # Merge strings with corresponding integers
    result = []
    for lst, string in zip(lists, strings):
        for item in lst:
            result.append([item, string])

    percentages = []

    for i, (idx, col) in enumerate(result):
        actual_value = operator_df.loc[idx, col]
        imputed_values = []
        if idx == 646:
            continue
        for df in dfs:
            try:
                imputed_values.append(df.loc[idx, col])
            except KeyError:
                continue

        # result has row indexes of rows with operators of original df. dfs, however, only stores imputed values after
        # removal of rows w/o target or too many missing values.
        if not len(imputed_values) == 0:
            print(f'{idx}: Actual value {actual_value}, imputed values: {imputed_values}')
            perc = get_percentage(imputed_values, actual_value)
            percentages.append(perc)

    print('Percentage average: ', np.mean(percentages))


if __name__ == '__main__':
    model_name = 'light_gbm'
    classification_type = 'cirrhosis'

    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option('display.max_rows', None)

    df = pd.read_excel(
        '../data/20231129 Lap und Histo Daten von Ines Tuschner.xlsx')
    df2 = pd.read_excel(
        '../data/202403 Lap und Histo Daten von Ines Tuschner.xlsx')
    df2 = df2[['HbA1c (%)', 'Glucose in plasma (mg/dL)',
               'LDL- Cholesterin (mg/dL)']]
    df = pd.concat([df, df2], axis=1)
    df_imp = df.copy()
    df = calculate_age(df)

    cols = [
        'Thrombozyten (Mrd/l)',
        'MCV (fl)',
        'Quick (%)',
        'INR',
        'Glucose in plasma (mg/dL)',
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
        'Patientennummer',

        'Blutentnahme',

        'Micro',
    ]

    df = df[cols]

    print_missing_y(df)
    print_high_missing_data(df)
    print_keep_samples_3_months(df)
    print_doubled_patients(df)

    true_df, dfs, mask, true_means, true_stds, columns, (indices, values, operator_cols) = imitate_preprocess(df_imp, classification_type=classification_type)
    means_std_per_feature = get_imputation_uncertainty(dfs, mask, columns)
    print('Missing percentages: \n', mask.mean() * 100)

    # Two-sample t-test
    # Assume data is normal
    for column in columns:
        if len(means_std_per_feature[column][0]) == 0:
            # Filtering out all unused columns
            continue
        x = np.array(true_df[column])
        x = x[~np.isnan(x)]
        t_stat, p_value = stats.ttest_ind(x, means_std_per_feature[column][0])
        print(
            f'Column: {column}, T-statistic: {t_stat}, True Mean: {true_means[column]}, True Std: {true_stds[column]}, Imp Mean: {np.mean(means_std_per_feature[column][0])}, Imp Std: {np.std(means_std_per_feature[column][1])}, P-value: {p_value} ')

    print('\n\n Calculating differences in y difference')
    true_df_0, _, _, true_means_0, true_stds_0, columns, (_, _, _) = imitate_preprocess(df_imp, filter=0, classification_type=classification_type)
    true_df_1, _, _, true_means_1, true_stds_1, columns, (_, _, _) = imitate_preprocess(df_imp, filter=1, classification_type=classification_type)
    true_df, _, _, true_means, true_stds, columns, (_, _, _) = imitate_preprocess(df_imp, classification_type=classification_type)

    print('Length of 0 df: ', len(true_df_0))
    print('Length of 1 df: ', len(true_df_1))

    # Two-sample t-test
    # Assume data is normal
    for column in columns:
        if column == 'Micro':
            continue
        x_0 = np.array(true_df_0[column])
        x_0 = x_0[~np.isnan(x_0)]
        x_1 = np.array(true_df_1[column])
        x_1 = x_1[~np.isnan(x_1)]

        t_stat, p_value = stats.ttest_ind(x_0, x_1)
        print(
            f'Column: {column}, T-statistic: {t_stat}, DF 0 Mean: {true_means_0[column]}, Std 0 : {true_stds_0[column]}, DF 1 Mean: {true_means_1[column]}, 1 Std: {true_stds_1[column]}, P-value: {p_value},')

    print('Compare operator with imputation')
    compare_operator_with_imputation(dfs, indices, values, operator_cols)

    dfs = []
    fib4_preds = []
    for i in range(10):
        df = pd.read_csv(f'../data/preprocessed_mice_fib_test/test_{classification_type}_{i}.csv')
        fib_4 = calculate_fib4(df, classification_type)
        fib_4 = fib_4['Fib4 Stages']
        fib4_preds.append(fib_4)
        dfs.append(df)

    fib4_preds = [df.tolist() for df in fib4_preds]

    file_path = f'outputs/{model_name}/{model_name}_{classification_type}_ensemble_preds.txt'

    # Initialize an empty list to hold the lists
    model_preds = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Use ast.literal_eval to safely evaluate the string as a Python expression
            model_preds.append(ast.literal_eval(line.strip()))

    accuracies1 = []
    accuracies2 = []
    f1s1 = []
    f1s2 = []
    precisions1 = []
    precisions2 = []
    recalls1 = []
    recalls2 = []

    for y_model1, y_model2, df in zip(model_preds, fib4_preds, dfs):
        y_true = df['Micro']
        print('Cohen cappa: ', cohen_kappa_score(y_model1, y_model2))

        accuracy_model1 = accuracy_score(y_true, y_model1)
        accuracy_model2 = accuracy_score(y_true, y_model2)

        precision_model1, recall_model1, _, _ = precision_recall_fscore_support(y_true, y_model1, average='macro')
        precision_model2, recall_model2, _, _ = precision_recall_fscore_support(y_true, y_model2, average='macro')

        # Calculate F1 score
        f1_model1 = f1_score(y_true, y_model1)
        f1_model2 = f1_score(y_true, y_model2)

        # Print accuracy and F1 scores
        print(f'Accuracy of Model 1: {accuracy_model1:.2f}')
        print(f'F1 Score of Model 1: {f1_model1:.2f}')

        print(f'Accuracy of FIB-4: {accuracy_model2:.2f}')
        print(f'F1 Score of FIB-4: {f1_model2:.2f}')

        # Constructing the contingency table for McNemar's test
        n01 = np.sum((y_model1 == y_true) & (y_model2 != y_true))  # Model 1 correct, Model 2 incorrect
        n10 = np.sum((y_model1 != y_true) & (y_model2 == y_true))  # Model 1 incorrect, Model 2 correct

        # Running McNemar's test
        result = mcnemar([[0, n01], [n10, 0]],
                         exact=True)  # Use exact=True for small sample sizes, exact=False for large

        print('Statistic:', result.statistic)
        print('p-value:', result.pvalue)

        accuracies1.append(accuracy_model1)
        accuracies2.append(accuracy_model2)

        precisions1.append(precision_model1)
        precisions2.append(precision_model2)

        recalls1.append(recall_model1)
        recalls2.append(recall_model2)

        f1s1.append(f1_model1)
        f1s2.append(f1_model2)

        print('Precision: ', precision_model2)
        print('Recall: ', recall_model2)
        print('\n ------------------------ \n')

    print(
        f'{model_name}\nAvg accuracy: {np.mean(accuracies1)},  std Accuracy: {np.std(accuracies2)} \nAvg F1: {np.mean(f1s1)}, Std F1 score: {np.std(f1s2)} \nAvg Precision: {np.mean(precisions1)}, Std Precision: {np.std(precisions2)}\n Std Recall {np.std(recalls2)}, Avg Recall: {np.mean(recalls1)} \n\n')
    print(
        f'FIB4\nAvg accuracy: {np.mean(accuracies2)}, std Accuracy: {np.std(accuracies2)} \nAvg F1: {np.mean(f1s2)} Std F1 score: {np.std(f1s2)} \nAvg Precision: {np.mean(precisions2)}, Std Precision: {np.std(precisions2)}\n Std Recall {np.std(recalls2)}, Avg Recall: {np.mean(recalls2)} \n\n')

    print(precisions2)
    print(recalls2)
