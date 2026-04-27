from pathlib import Path

import pandas as pd

from lab_window_robustness import choose_recommended_window


if __name__ == '__main__':
    csv_path = Path('../outputs/robustness/lab_window_auroc_comparison.csv')
    df = pd.read_csv(csv_path)
    recommendation = choose_recommended_window(df)

    print('Recommended final lab window (highest AUROC with cross-cohort stability):')
    print(
        f"{recommendation['window_label']} | "
        f"internal={recommendation['internal_auroc']:.4f}, "
        f"external={recommendation['external_auroc']:.4f}, "
        f"mean={recommendation['mean_auroc']:.4f}, "
        f"gap={recommendation['cohort_gap']:.4f}"
    )
