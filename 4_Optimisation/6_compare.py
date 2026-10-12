"""
COMPARISON OF THE OPTIMISED LINEAR REGRESSION MODELS

This script combines the results saved by the earlier scripts (original, log,
z-score and Yeo-Johnson) into one table and saves it as optimisation_results.csv. 
For each of Models 2.1 and 2.2, it prints the training fit and test-set metrics 
of every version on both splits. 
It then picks the best version as the one with the lowest average test RMSE across
the 80/20 and 70/30 splits.
"""

# To import necesaary libraries and modules
import os
import pandas as pd

# Folder of this script and the results folder
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

# Results saved by scripts 1, 2, 4 and 5
FILES = ["original.csv", "log.csv", "zscore.csv", "yeojohnson.csv"]
results = pd.concat([pd.read_csv(os.path.join(OUT, f), dtype={"model": str}) for f in FILES], ignore_index=True)
results.round(4).to_csv(os.path.join(OUT, "optimisation_results.csv"), index=False)

pd.set_option("display.width", 200)
show = ["split", "version", "train_R2", "train_adj_R2", "AIC", "cond_no",
        "test_MAE", "test_RMSE", "test_RMSE_norm", "test_R2"]
for name in ["2.1", "2.2"]:
    r = results[results["model"] == name].sort_values(["split", "version"], ascending=[False, True])
    print("=" * 78 + f"\nLINEAR REGRESSION {name}: COMPARISON OF OPTIMISATIONS\n" + "=" * 78)
    print(r[show].round(4).to_string(index=False))

    # Version with the lowest average test RMSE over the two splits
    avg = r.groupby("version")["test_RMSE"].mean().sort_values()
    print("\nAverage test RMSE over both splits (lower is better):")
    print(avg.round(4).to_string())
    print(f"Best version for Model {name}: {avg.index[0]}\n")


print(f"Saved {os.path.join('4_Optimisation', 'results', 'optimisation_results.csv')}")