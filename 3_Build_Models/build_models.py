import os
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Folder of this script and the project folder
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SPLITS = ["Set_80_20", "Set_70_30"]


def equation(y_col, intercept, coefs, x_cols):
    """Write the full multiple linear regression equation."""
    terms = " ".join(f"{'+' if c >= 0 else '-'} {abs(c):.4f} ({v})" for c, v in zip(coefs, x_cols))
    return f"{y_col} = {intercept:.4f} {terms}"


def evaluate(name, y_test, y_pred, y):
    """Compute and print the standard performance metrics."""
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    rmse_norm = rmse / (y.max() - y.min())        # normalised by the range of y
    r_2 = metrics.r2_score(y_test, y_pred)
    print(f"\n{name} performance:")
    print("MAE: ", round(mae, 4))
    print("MSE: ", round(mse, 4))
    print("RMSE: ", round(rmse, 4))
    print("RMSE (Normalised): ", round(rmse_norm, 4))
    print("R^2: ", round(r_2, 4))
    return dict(MAE=mae, MSE=mse, RMSE=rmse, RMSE_norm=rmse_norm, R2=r_2)


def build_and_evaluate(title, file, x_cols, y_col, id_cols, final_file):
    """function define to Build a linear regression on each split, evaluate it on the test set and compare it with the mean baseline."""
    # Read dataset into a DataFrame
    df = pd.read_csv(os.path.join(ROOT, "1_Build_Datasets", file))

    # Save the final dataset: ID fields, exactly 8 explanatory variables, response variable and split labels
    df[id_cols + x_cols + [y_col] + SPLITS].to_csv(os.path.join(HERE, final_file), index=False)
    print(f"Saved 3_Build_Models/{final_file}: {len(df)} rows, {len(x_cols)} explanatory variables")

    y = df[y_col].values
    results = []

    for split in SPLITS:
        print("\n" + "=" * 70)
        print(f"{title} | {split}")
        print("=" * 70)

        # Training and test rows come from the split column
        train = df[df[split] == "Train"]
        test = df[df[split] == "Test"]

        # Separate explanatory variables (x) from the response variable (y)
        X_train, y_train = train[x_cols].values, train[y_col].values
        X_test, y_test = test[x_cols].values, test[y_col].values
        print(f"Training rows: {len(train)}   Test rows: {len(test)}")

        # ---------------- BUILD AND EVALUATE A LINEAR REGRESSION MODEL

        # Build a linear regression model
        model = LinearRegression()

        # Train (fit) the linear regression model using the training set
        model.fit(X_train, y_train)

        # Print the intercept and coefficients learned by the linear regression model
        print("\nIntercept: ", round(model.intercept_, 4))
        print("Coefficients:")
        print(pd.Series(model.coef_, index=x_cols).round(4).to_string())
        print("\nRegression equation:")
        print(equation(y_col, model.intercept_, model.coef_, x_cols))

        # Use linear regression to predict the values of (y) in the test set
        y_pred = model.predict(X_test)

        # Show the predicted values of (y) next to the actual values of (y)
        df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.round(2)})
        print(df_pred.to_string())

        lr = evaluate("Linear regression", y_test, y_pred, y)

        # ---------------- COMPARE WITH A DUMMY MODEL (BASELINE) THAT USES THE MEAN AS ITS PREDICTION

        # Compute mean of values in (y) training set
        y_base = np.mean(y_train)

        # Replicate the mean value as many times as there are values in the test set
        y_pred_base = [y_base] * len(y_test)

        base = evaluate("Baseline (mean)", y_test, y_pred_base, y)

        results.append(dict(split=split, model="Linear regression", **lr))
        results.append(dict(split=split, model="Baseline (mean)", **base))

    summary = pd.DataFrame(results)
    print(f"\n##### SUMMARY: {title} #####")
    print(summary.round(4).to_string(index=False))
    return summary


# ==================================================================== LINEAR REGRESSION 2.1
# Predicts the goal difference (Team A goals - Team B goals) of a match; 104 rows, one per match
summary_21 = build_and_evaluate(
    title="LINEAR REGRESSION 2.1 (goal difference)",
    file="LR2.1_dataset.csv",
    x_cols=["fifa_diff", "scored_diff", "conceded_diff", "xg_diff",
            "ppm_diff", "attack_out_diff", "host", "h2h_gd"],
    y_col="goal_diff",
    id_cols=["match_no", "date", "stage", "team_A", "team_B"],
    final_file="LR2.1_final.csv",
)

# ==================================================================== LINEAR REGRESSION 2.2
# Predicts the number of goals scored by a team in a match; 208 rows, one per team per match
summary_22 = build_and_evaluate(
    title="LINEAR REGRESSION 2.2 (goals scored by a team)",
    file="LR2.2_dataset.csv",
    x_cols=["fifa_diff", "score_form", "opp_concede_form", "xg_for",
            "opp_xga", "attack_p90", "host", "fail_to_score"],
    y_col="goals",
    id_cols=["match_no", "date", "stage", "team", "opponent", "side"],
    final_file="LR2.2_final.csv",
)

# Save both summaries in one file
pd.concat([summary_21.assign(lr="2.1"), summary_22.assign(lr="2.2")]).round(4).to_csv(
    os.path.join(HERE, "model_results.csv"), index=False)
print("\nSaved 3_Build_Models/model_results.csv")