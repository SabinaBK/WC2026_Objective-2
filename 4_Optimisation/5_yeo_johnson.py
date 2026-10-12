"""
YEO-JOHNSON POWER TRANSFORMATION OF MODELS 2.1 AND 2.2

This script applies the Yeo-Johnson power transformation to the explanatory variables of 
each model to make their distributions more Gaussian-like. 
Unlike the log, Yeo-Johnson also works on zero and negative values, so it can be applied to Model 2.1. 
The transformer is fitted on the training rows only.
The script then re-fits each model with statsmodels OLS on the 80/20 and 70/30
splits and evaluates it on the test rows. 
For the 80/20 split, it compares the skewness and histograms of each variable before and after
the transformation.
The test-set results are saved to the results folder.
"""

# To import necesaary libraries and modules
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import PowerTransformer

# Folder of this script, the project folder (datasets) and the results folder
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "3_Build_Models")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

SPLITS = ["Set_80_20", "Set_70_30"]
MODELS = {
    "2.1": dict(file="LR2.1_final.csv", y_col="goal_diff",
                x_cols=["fifa_diff", "scored_diff", "conceded_diff", "xg_diff",
                        "ppm_diff", "attack_out_diff", "host", "h2h_gd"]),
    "2.2": dict(file="LR2.2_final.csv", y_col="goals",
                x_cols=["fifa_diff", "score_form", "opp_concede_form", "xg_for",
                        "opp_xga", "attack_p90", "host", "fail_to_score"]),
}


def evaluate(model, X_test, y_test, y_all):
    """Training fit statistics and test-set performance metrics."""
    y_pred = model.predict(sm.add_constant(X_test, has_constant="add"))
    rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    return dict(train_R2=model.rsquared, train_adj_R2=model.rsquared_adj, AIC=model.aic, BIC=model.bic,
                cond_no=model.condition_number, test_MAE=metrics.mean_absolute_error(y_test, y_pred),
                test_MSE=metrics.mean_squared_error(y_test, y_pred), test_RMSE=rmse,
                test_RMSE_norm=rmse / (y_all.max() - y_all.min()), test_R2=metrics.r2_score(y_test, y_pred))


results = []
for name, cfg in MODELS.items():
    df = pd.read_csv(os.path.join(DATA, cfg["file"]))
    x_cols, y_col = cfg["x_cols"], cfg["y_col"]

    for split in SPLITS:
        train, test = df[df[split] == "Train"], df[df[split] == "Test"]

        """
        APPLY POWER TRANSFORMER (YEO-JOHNSON)
        """
        # Fit the transformer on the training set only, then apply it to both sets
        pt = PowerTransformer(method="yeo-johnson").fit(train[x_cols])
        X_train = pd.DataFrame(pt.transform(train[x_cols]), index=train.index, columns=x_cols)
        X_test = pd.DataFrame(pt.transform(test[x_cols]), index=test.index, columns=x_cols)

        """
        REBUILD AND REEVALUATE LINEAR REGRESSION USING MORE GAUSSIAN-LIKE EXPLANATORY VARIABLES
        """
        model = sm.OLS(train[y_col], sm.add_constant(X_train)).fit()
        results.append(dict(model=name, split=split, version="Yeo-Johnson",
                            **evaluate(model, X_test, test[y_col], df[y_col])))

        if split == "Set_80_20":
            print("=" * 78 + f"\nLINEAR REGRESSION {name} | {split} | Yeo-Johnson\n" + "=" * 78)
            print(model.summary())

            # Skewness of each variable before and after the transformation
            skew = pd.DataFrame({"before": [stats.skew(train[v]) for v in x_cols],
                                 "after": [stats.skew(X_train[v]) for v in x_cols]}, index=x_cols)
            print("\nSkewness before and after Yeo-Johnson:")
            print(skew.round(2).to_string())
            print()

            # Distribution of each variable before and after the transformation
            fig, axes = plt.subplots(len(x_cols), 2, figsize=(8, 2 * len(x_cols)))
            for i, v in enumerate(x_cols):
                axes[i, 0].hist(train[v], bins=15, color="#2a78d6")
                axes[i, 0].set_title(f"{v} (original, skew {skew.loc[v, 'before']:.2f})", fontsize=9)
                axes[i, 1].hist(X_train[v], bins=15, color="#1baf7a")
                axes[i, 1].set_title(f"{v} (Yeo-Johnson, skew {skew.loc[v, 'after']:.2f})", fontsize=9)
            fig.suptitle(f"Model {name}: distributions before and after Yeo-Johnson")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"yeo_johnson_{name}.png"), dpi=110)
            plt.show()

results = pd.DataFrame(results)
results.round(4).to_csv(os.path.join(OUT, "yeojohnson.csv"), index=False)
print("\nYEO-JOHNSON MODELS")
print(results.round(4).to_string(index=False))