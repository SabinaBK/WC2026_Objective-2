"""
COMPARISON OF THE ORIGINAL AND BACKUP VARIABLE SETS

This script runs test_variables.py for every model (2.1 and 2.2), variable set
(original and backup) and split (80/20 and 70/30). 
It then compares the backup set with the original set on each split. 
The original set has a problem if the variable being replaced has the wrong coefficient sign 
or the highest VIF is above 10. 
The backup set passes if the added variable has the right sign, the highest VIF is 10 or below,
and the cross-validated R^2 falls by no more than 0.02. 
A variable is swapped only when the original set has a problem and the
backup set passes on both splits. 
All results are saved in test_results.xlsx.
"""

# To import necesaary libraries and modules
import os
import subprocess
import sys
import pandas as pd

# Folder of this script
HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = ["2.1", "2.2"]
SETS = ["original", "backup"]
SPLITS = ["Set_80_20", "Set_70_30"]
SWAPPED = {"2.1": ("ucl_min_diff", "ppm_diff"), "2.2": ("opp_ucl_def", "fail_to_score")}
MAX_CV_LOSS = 0.02   # largest allowed fall in cross-validated R^2 when swapping

# Run test_variables.py for every model, variable set and split
summary, details = [], {}
for model in MODELS:
    for s in SETS:
        for split in SPLITS:
            run = subprocess.run([sys.executable, os.path.join(HERE, "test_variables.py"), model, split, s],
                                 capture_output=True, text=True)
            if run.returncode != 0:
                sys.exit(f"test_variables.py {model} {split} {s} failed:\n{run.stderr}")
            folder = os.path.join(HERE, "tests", f"{model}_{s}_{split}")
            summary.append(pd.read_csv(os.path.join(folder, "model_summary.csv"), dtype={"model": str}))
            details[f"{model}_{s}_{split[4:]}"] = pd.read_csv(os.path.join(folder, "variable_tests.csv"), index_col=0)

summary = pd.concat(summary, ignore_index=True)


def var_row(model, s, split, var):
    """Test results of one variable in one run."""
    return details[f"{model}_{s}_{split[4:]}"].loc[var]


def model_row(model, s, split):
    """Whole-model results of one run."""
    return summary[(summary["model"] == model) & (summary["set"] == s) & (summary["split"] == split)].iloc[0]


# Compare the backup set with the original set on each split
decision = []
for model in MODELS:
    old, new = SWAPPED[model]
    for split in SPLITS:
        o, b = model_row(model, "original", split), model_row(model, "backup", split)
        removed_ok = bool(var_row(model, "original", split, old)["sign_ok"])
        added_ok = bool(var_row(model, "backup", split, new)["sign_ok"])
        decision.append(dict(
            model=model, split=split, removed=old, added=new,
            cv_r2_original=o["cv_r2"], cv_r2_backup=b["cv_r2"],
            max_vif_original=o["max_vif"], max_vif_backup=b["max_vif"],
            removed_sign_ok=removed_ok, added_sign_ok=added_ok,
            original_has_problem=(not removed_ok) or o["max_vif"] > 10,
            backup_passes=added_ok and b["max_vif"] <= 10 and b["cv_r2"] - o["cv_r2"] >= -MAX_CV_LOSS))
decision = pd.DataFrame(decision)

# Save every result in one workbook
with pd.ExcelWriter(os.path.join(HERE, "test_results.xlsx")) as xw:
    summary.round(3).to_excel(xw, sheet_name="Summary", index=False)
    decision.round(3).to_excel(xw, sheet_name="Decision", index=False)
    for name, d in details.items():
        d.round(3).to_excel(xw, sheet_name=name[:31])

pd.set_option("display.width", 200)
print("SUMMARY\n" + summary.round(3).to_string(index=False))
print("\nDECISION (backup compared with original on each split)")
print(decision.round(3).to_string(index=False))
print()
for model in MODELS:
    d = decision[decision["model"] == model]
    old, new = SWAPPED[model]
    if d["original_has_problem"].any() and d["backup_passes"].all():
        print(f"Model {model}: SWAP {old} -> {new} (original has a problem and the backup passes on both splits)")
    elif not d["original_has_problem"].any():
        print(f"Model {model}: KEEP {old} (no problem on either split)")
    else:
        print(f"Model {model}: KEEP {old} (backup does not pass on both splits)")
print("\nSaved test_results.xlsx and the tests folder in 2_Test_Variables.")