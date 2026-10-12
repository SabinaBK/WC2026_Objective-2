"""
MULTICOLLINEARITY CHECK OF MODELS 2.1 AND 2.2

This script checks whether the explanatory variables of each model are too
strongly related to each other, using the 80/20 training rows. 
It plots a correlation heatmap of the explanatory and response variables, lists the three
most strongly correlated pairs of explanatory variables, and calculates the
variance inflation factor (VIF) of each one, where below 5 is fine and above
10 signals a problem.
"""

# To import necesaary libraries and modules
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Folder of this script, the project folder (datasets) and the results folder
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "3_Build_Models")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "2.1": dict(file="LR2.1_final.csv", y_col="goal_diff",
                x_cols=["fifa_diff", "scored_diff", "conceded_diff", "xg_diff",
                        "ppm_diff", "attack_out_diff", "host", "h2h_gd"]),
    "2.2": dict(file="LR2.2_final.csv", y_col="goals",
                x_cols=["fifa_diff", "score_form", "opp_concede_form", "xg_for",
                        "opp_xga", "attack_p90", "host", "fail_to_score"]),
}

for name, cfg in MODELS.items():
    df = pd.read_csv(os.path.join(DATA, cfg["file"]))
    x_cols, y_col = cfg["x_cols"], cfg["y_col"]
    train = df[df["Set_80_20"] == "Train"]

    # Correlation matrix of the explanatory variables and the response variable
    corr = train[x_cols + [y_col]].corr()

    # Plot the pairwise correlation as heatmap
    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
                     square=False, annot=True, fmt=".2f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    plt.title(f"Model {name}: correlation matrix (80/20 training rows)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"corr_matrix_{name}.png"), dpi=130)
    plt.show()

    # Strongest correlations between pairs of explanatory variables
    pairs = corr.loc[x_cols, x_cols].abs().where(~np.eye(len(x_cols), dtype=bool)).stack()
    pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)].sort_values(ascending=False)

    # Variance inflation factor of each explanatory variable
    X = sm.add_constant(train[x_cols])
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=x_cols)

    print("=" * 78 + f"\nLINEAR REGRESSION {name} | multicollinearity (80/20 training rows)\n" + "=" * 78)
    print("Strongest correlations between explanatory variables:")
    print(pairs.head(3).round(2).to_string())
    print("\nVIF (< 5 fine, > 10 problem):")
    print(vif.round(2).to_string())
    print(f"\nHighest |r| = {pairs.iloc[0]:.2f}, highest VIF = {vif.max():.2f}\n")