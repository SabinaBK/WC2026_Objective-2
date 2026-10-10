import os
import math
import numpy as np
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

# Only Model 2.2: every 2.1 variable is a Team A - Team B difference with negative values, so log is not possible
FILE, Y_COL = "LR2.2_final.csv", "goals"
X_COLS = ["fifa_diff", "score_form", "opp_concede_form", "xg_for",
          "opp_xga", "attack_p90", "host", "fail_to_score"]
LOG_COLS = ["xg_for", "score_form"]         # positive variables with a curved relationship to goals


def evaluate(model, X_test, y_test, y_all):
    """function evaluate is defined here forTraining fit statistics and test-set performance metrics."""
    y_pred = model.predict(sm.add_constant(X_test, has_constant="add"))
    rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    return dict(train_R2=model.rsquared, train_adj_R2=model.rsquared_adj, AIC=model.aic, BIC=model.bic,
                cond_no=model.condition_number, test_MAE=metrics.mean_absolute_error(y_test, y_pred),
                test_MSE=metrics.mean_squared_error(y_test, y_pred), test_RMSE=rmse,
                test_RMSE_norm=rmse / (y_all.max() - y_all.min()), test_R2=metrics.r2_score(y_test, y_pred))


print("Model 2.1: log transformation not applied (all variables take negative values).\n")

# Read dataset into a DataFrame
df = pd.read_csv(os.path.join(DATA, FILE))

"""
NON-LINEAR TRANSFORMATION OF THE SELECTED VARIABLES
"""
df_log = df.copy()
for v in LOG_COLS:
    df_log[v] = np.log(df_log[v])

# Visualise the effect of the transformation (80/20 training rows)
train = df[df["Set_80_20"] == "Train"]
fig, axes = plt.subplots(len(LOG_COLS), 2, figsize=(10, 3.6 * len(LOG_COLS)))
for row, v in zip(axes, LOG_COLS):
    row[0].scatter(train[v], train[Y_COL], color="green", s=15)
    row[0].set_title(f"Original {v}"); row[0].set_xlabel(v); row[0].set_ylabel(Y_COL)
    row[1].scatter(np.log(train[v]), train[Y_COL], color="red", s=15)
    row[1].set_title(f"Log transformed {v}"); row[1].set_xlabel(f"log({v})"); row[1].set_ylabel(Y_COL)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "log_transform_2.2.png"), dpi=130)
plt.show()

"""
RE-RUN THE LINEAR REGRESSION MODEL WITH THE TRANSFORMED VARIABLES
"""
results = []
for split in SPLITS:
    train, test = df_log[df_log[split] == "Train"], df_log[df_log[split] == "Test"]
    model = sm.OLS(train[Y_COL], sm.add_constant(train[X_COLS])).fit()
    results.append(dict(model="2.2", split=split, version="Log",
                        **evaluate(model, test[X_COLS], test[Y_COL], df[Y_COL])))
    if split == "Set_80_20":
        print("=" * 78 + f"\nLINEAR REGRESSION 2.2 | {split} | log({', '.join(LOG_COLS)})\n" + "=" * 78)
        print(model.summary())

results = pd.DataFrame(results)
results.round(4).to_csv(os.path.join(OUT, "log.csv"), index=False)
print("\nLOG-TRANSFORMED MODEL 2.2")
print(results.round(4).to_string(index=False))