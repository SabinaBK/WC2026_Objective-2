"""
BUILD AND EVALUATE THE ORIGINAL MULTIPLE LINEAR REGRESSION MODELS
 
This script fits Model 2.1 (goal difference) and Model 2.2 (goals scored)
with statsmodels OLS on the training rows of two train/test splits
(80/20 and 70/30), evaluates each model on its test rows, plots each
explanatory variable against the response variable to inspect linearity,
and saves the results to the results folder.
"""

# To import necesaary libraries and modules
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics

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


def equation(y_col, params):
    """Write the full multiple linear regression equation from statsmodels parameters."""
    terms = " ".join(f"{'+' if c >= 0 else '-'} {abs(c):.4f} ({v})" for v, c in params.drop("const").items())
    return f"{y_col} = {params['const']:.4f} {terms}"


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

    """
    BUILD AND EVALUATE LINEAR REGRESSION USING STATSMODELS
    """
    for split in SPLITS:
        train, test = df[df[split] == "Train"], df[df[split] == "Test"]

        # Build the linear regression model on the training set
        model = sm.OLS(train[y_col], sm.add_constant(train[x_cols])).fit()
        results.append(dict(model=name, split=split, version="Original",
                            **evaluate(model, test[x_cols], test[y_col], df[y_col])))

        if split == "Set_80_20":
            print("=" * 78 + f"\nLINEAR REGRESSION {name} | {split} | statsmodels\n" + "=" * 78)
            print(model.summary())
            print("\nRegression equation:")
            print(equation(y_col, model.params))

    """
    INSPECT THE LINEARITY OF EACH EXPLANATORY VARIABLE WITH THE RESPONSE VARIABLE
    """
    train = df[df["Set_80_20"] == "Train"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, v in zip(axes.ravel(), x_cols):
        ax.scatter(train[v], train[y_col], color="#2a78d6", s=15)
        ax.set_xlabel(v)
        ax.set_ylabel(y_col)
    fig.suptitle(f"Model {name}: each explanatory variable vs {y_col} (80/20 training rows)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"linearity_{name}.png"), dpi=130)
    plt.show()

results = pd.DataFrame(results)
results.round(4).to_csv(os.path.join(OUT, "original.csv"), index=False)
print("\nORIGINAL MODELS (fitted on training rows, evaluated on test rows)")
print(results.round(4).to_string(index=False))