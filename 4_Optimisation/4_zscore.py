import os
import math
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

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
        APPLY Z-SCORE STANDARDISATION
        """
        # Fit the scaler on the training set only, then apply it to both sets
        scaler = StandardScaler().fit(train[x_cols])
        X_train = pd.DataFrame(scaler.transform(train[x_cols]), index=train.index, columns=x_cols)
        X_test = pd.DataFrame(scaler.transform(test[x_cols]), index=test.index, columns=x_cols)

        """
        REBUILD AND REEVALUATE LINEAR REGRESSION USING STANDARDISED EXPLANATORY VARIABLES
        """
        model = sm.OLS(train[y_col], sm.add_constant(X_train)).fit()
        results.append(dict(model=name, split=split, version="Z-score",
                            **evaluate(model, X_test, test[y_col], df[y_col])))

        if split == "Set_80_20":
            print("=" * 78 + f"\nLINEAR REGRESSION {name} | {split} | z-score\n" + "=" * 78)
            print(model.summary())
            # Standardised coefficients: change in y for a 1 standard deviation increase in each variable
            print("\nStandardised coefficients, largest effect first:")
            print(model.params.drop("const").sort_values(key=abs, ascending=False).round(4).to_string())
            print()

results = pd.DataFrame(results)
results.round(4).to_csv(os.path.join(OUT, "zscore.csv"), index=False)
print("\nZ-SCORE MODELS")
print(results.round(4).to_string(index=False))