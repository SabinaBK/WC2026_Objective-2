"""
VARIABLE TESTING FOR THE LINEAR REGRESSION DATASETS

This script tests the explanatory variables of Model 2.1 (LR2.1_dataset.csv)
or Model 2.2 (LR2.2_dataset.csv) on the training rows of the chosen split,
for either the original or the backup variable set (the backup swaps one
variable). 
The tests include a data check, correlation with the response variable, 
scatter plots with a line of best fit, standardised coefficients checked against 
the expected sign, variance inflation factor (VIF), a drop-one
test (5-fold cross-validated R^2) and distribution/skewness before and after
Yeo-Johnson. 
Each variable then gets a verdict of KEEP, REVIEW or SWAP for
backup. 
The results and charts are saved to the tests folder.

Usage: python test_variables.py <2.1|2.2> <Set_80_20|Set_70_30> <original|backup>
"""
# To import necesaary libraries and modules
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# Usage: python test_variables.py <2.1|2.2> <Set_80_20|Set_70_30> <original|backup>
MODEL = sys.argv[1] if len(sys.argv) > 1 else "2.1"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "Set_80_20"
SET = sys.argv[3] if len(sys.argv) > 3 else "original"

# Folder of this script and the project folder
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Dataset file and response variable of each model
FILES = {"2.1": os.path.join(ROOT, "1_Build_Datasets", "LR2.1_dataset.csv"),
         "2.2": os.path.join(ROOT, "1_Build_Datasets", "LR2.2_dataset.csv")}
Y_COLS = {"2.1": "goal_diff", "2.2": "goals"}

# Expected sign of each variable's effect on Y
SIGNS = {
    "2.1": {"fifa_diff": +1, "scored_diff": +1, "conceded_diff": -1, "xg_diff": +1,
            "ucl_min_diff": +1, "attack_out_diff": +1, "host": +1, "h2h_gd": +1, "ppm_diff": +1},
    "2.2": {"fifa_diff": +1, "score_form": +1, "opp_concede_form": +1, "xg_for": +1,
            "opp_xga": +1, "attack_p90": +1, "host": +1, "opp_ucl_def": -1, "fail_to_score": -1},
}
# Original set and backup set (the backup swaps one variable)
SETS = {
    "2.1": {"original": ["fifa_diff", "scored_diff", "conceded_diff", "xg_diff",
                         "ucl_min_diff", "attack_out_diff", "host", "h2h_gd"],
            "backup":   ["fifa_diff", "scored_diff", "conceded_diff", "xg_diff",
                         "ppm_diff", "attack_out_diff", "host", "h2h_gd"]},
    "2.2": {"original": ["fifa_diff", "score_form", "opp_concede_form", "xg_for",
                         "opp_xga", "attack_p90", "host", "opp_ucl_def"],
            "backup":   ["fifa_diff", "score_form", "opp_concede_form", "xg_for",
                         "opp_xga", "attack_p90", "host", "fail_to_score"]},
}
CONFIG = dict(file=FILES[MODEL], y=Y_COLS[MODEL],
              expected={v: SIGNS[MODEL][v] for v in SETS[MODEL][SET]})

# Thresholds
R_STRONG, R_WEAK = 0.3, 0.1
VIF_WARN, VIF_BAD = 5, 10
SKEW_LIMIT = 1.0

out = os.path.join(HERE, "tests", f"{MODEL}_{SET}_{SPLIT}")
os.makedirs(out, exist_ok=True)

# ------------------------------------------------------------------ 1. data check
df = pd.read_csv(CONFIG["file"])
y_col, expected = CONFIG["y"], CONFIG["expected"]
train = df[(df[SPLIT] == "Train") & df[y_col].notna()].copy()
print(f"Model {MODEL} | set {SET} | split {SPLIT} | training rows with Y: {len(train)}")

print("\nTEST 1 - data check (training rows)")
usable = []
for v in expected:
    n_miss = train[v].isna().sum()
    if n_miss == len(train):
        print(f"  {v:18s} SKIPPED - no data")
    elif train[v].nunique() <= 1:
        print(f"  {v:18s} SKIPPED - same value in every row")
    else:
        if n_miss:
            print(f"  {v:18s} {n_miss} missing value(s) - those rows are dropped")
        usable.append(v)
if not usable:
    sys.exit("No variables have data.")
train = train.dropna(subset=usable)
X, y = train[usable], train[y_col]
groups = train["match_no"]          # keeps both rows of a 2.2 match in the same CV fold
print(f"  Testing {len(usable)} variable(s) on {len(train)} complete training rows.")

res = pd.DataFrame(index=usable)
res["expected_sign"] = [expected[v] for v in usable]

# ------------------------------------------------------------------ 2. correlation with Y
r_p = [stats.pearsonr(X[v], y) for v in usable]
res["r"] = [a for a, _ in r_p]
res["r_p_value"] = [b for _, b in r_p]

# ------------------------------------------------------------------ 3. scatter plots
n = len(usable)
cols = 4
fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(4 * cols, 3.4 * np.ceil(n / cols)))
for ax, v in zip(np.ravel(axes), usable):
    ax.scatter(X[v], y, s=14, alpha=0.6, color="#2E5597")
    slope, intercept = np.polyfit(X[v], y, 1)                  # straight line of best fit
    xs = np.linspace(X[v].min(), X[v].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="#C0504D", lw=1.5)
    ax.set_title(f"{v}  (r = {res.loc[v, 'r']:.2f})", fontsize=10)
    ax.set_xlabel(v, fontsize=8); ax.set_ylabel(y_col, fontsize=8)
for ax in np.ravel(axes)[n:]:
    ax.axis("off")
fig.suptitle(f"Model {MODEL}: each variable vs {y_col} (training rows)")
fig.tight_layout(); fig.savefig(os.path.join(out, "scatter_vs_y.png"), dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 4. sign + standardised coefficients
scaler = StandardScaler().fit(X)                 # fitted on training rows only
Xs = pd.DataFrame(scaler.transform(X), columns=usable, index=X.index)
ols = sm.OLS(y, sm.add_constant(Xs)).fit()
res["std_coef"] = ols.params[usable].values
res["coef_p_value"] = ols.pvalues[usable].values
res["sign_ok"] = np.sign(res["std_coef"]) == res["expected_sign"]
res["corr_sign_ok"] = np.sign(res["r"]) == res["expected_sign"]
with open(os.path.join(out, "ols_summary.txt"), "w") as f:
    f.write(ols.summary().as_text())

# ------------------------------------------------------------------ 5. VIF
Xc = sm.add_constant(Xs)
res["VIF"] = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])] if n > 1 else 1.0

# ------------------------------------------------------------------ 6. drop-one test (5-fold CV R^2)
cv = GroupKFold(n_splits=5)
def cv_r2(cols_):
    return cross_val_score(LinearRegression(), Xs[cols_], y, groups=groups, cv=cv, scoring="r2").mean()
base_r2 = cv_r2(usable)
res["r2_drop_when_removed"] = [base_r2 - cv_r2([c for c in usable if c != v]) if n > 1 else base_r2
                               for v in usable]

# ------------------------------------------------------------------ 7. distribution
pt = PowerTransformer(method="yeo-johnson")
res["skew"] = [stats.skew(X[v]) for v in usable]
res["n_unique"] = [X[v].nunique() for v in usable]
res["skew_after_YJ"] = [stats.skew(pt.fit_transform(X[[v]]).ravel()) for v in usable]
fig, axes = plt.subplots(n, 2, figsize=(8, 2.2 * n))
axes = np.atleast_2d(axes)
for i, v in enumerate(usable):
    axes[i, 0].hist(X[v], bins=15, color="#2E5597")
    axes[i, 0].set_title(f"{v} (original, skew {res.loc[v, 'skew']:.2f})", fontsize=9)
    axes[i, 1].hist(pt.fit_transform(X[[v]]).ravel(), bins=15, color="#4F9D69")
    axes[i, 1].set_title(f"{v} (Yeo-Johnson, skew {res.loc[v, 'skew_after_YJ']:.2f})", fontsize=9)
fig.suptitle(f"Model {MODEL}: distributions (training rows)")
fig.tight_layout(); fig.savefig(os.path.join(out, "histograms.png"), dpi=110); plt.close(fig)

# ------------------------------------------------------------------ verdict
def verdict(row):
    corr_pass = abs(row["r"]) >= R_STRONG
    drop_pass = row["r2_drop_when_removed"] > 0
    notes = []
    if row["VIF"] > VIF_BAD:
        notes.append("VIF > 10: overlaps strongly with other variables")
    elif row["VIF"] > VIF_WARN:
        notes.append("VIF > 5: some overlap")
    if abs(row["skew"]) > SKEW_LIMIT and row["n_unique"] > 3:   # not for 0/1 indicators
        notes.append("skewed: consider Yeo-Johnson/log")
    if not row["sign_ok"] and row["corr_sign_ok"]:
        notes.append("coefficient sign flipped by other variables (collinearity)")
    if row["sign_ok"] and (corr_pass or drop_pass):
        v = "KEEP"
    elif row["corr_sign_ok"] and (abs(row["r"]) >= R_WEAK or drop_pass):
        v = "REVIEW"
    else:
        v = "SWAP for backup"
    if row.name == "host" and v != "KEEP":
        v = "KEEP (theory)"
        notes.append("few non-zero rows; kept because home advantage is well established")
    return pd.Series({"verdict": v, "notes": "; ".join(notes)})
res = res.join(res.apply(verdict, axis=1))

# ------------------------------------------------------------------ print + save
show = res[["r", "std_coef", "sign_ok", "VIF", "r2_drop_when_removed", "skew", "verdict"]].copy()
pd.set_option("display.width", 160)
print("\nRESULTS (training rows only)")
print(show.round(3).to_string())
print(f"\nFull-model cross-validated R^2 (5-fold): {base_r2:.3f}")
print(f"Full-model in-sample R^2: {ols.rsquared:.3f}   adjusted R^2: {ols.rsquared_adj:.3f}")
for v, row in res.iterrows():
    if row["notes"]:
        print(f"  note - {v}: {row['notes']}")
res.round(4).to_csv(os.path.join(out, "variable_tests.csv"))
pd.DataFrame([dict(model=MODEL, set=SET, split=SPLIT, rows=len(train), cv_r2=base_r2,
                   r2=ols.rsquared, adj_r2=ols.rsquared_adj, max_vif=res["VIF"].max(),
                   wrong_signs=int((~res["sign_ok"]).sum()),
                   keep=int(res["verdict"].str.startswith("KEEP").sum()),
                   review=int((res["verdict"] == "REVIEW").sum()),
                   swap=int(res["verdict"].str.startswith("SWAP").sum()))]
             ).round(4).to_csv(os.path.join(out, "model_summary.csv"), index=False)
print(f"\nSaved results and charts in {os.path.relpath(out, ROOT)}/")