"""
Script to programmatically generate all EDA and experiment notebooks.
Run from the project root:  python create_notebooks.py
"""

import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)
(ROOT / "models" / "credit_risk").mkdir(parents=True, exist_ok=True)
(ROOT / "models" / "network_intrusion").mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 1: Credit Risk EDA
# ─────────────────────────────────────────────────────────────────────────────

def make_credit_eda():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

    cells = [
        nbf.v4.new_markdown_cell(
            "# Credit Risk — Exploratory Data Analysis\n\n"
            "**Dataset**: UCI Default of Credit Card Clients  \n"
            "**Records**: 30,000 | **Features**: 23 | **Target**: `default` (1 = defaulted next month)\n\n"
            "## Why EDA First?\n"
            "EDA is not optional — it's the foundation of every good ML project. Before building a model:\n"
            "- We need to understand what each feature actually means\n"
            "- Spot data quality issues (wrong types, impossible values) early\n"
            "- Understand class imbalance — critical for risk prediction\n"
            "- Identify which features are likely predictive\n\n"
            "Skipping EDA leads to silent model failures that are very hard to debug later."
        ),

        nbf.v4.new_code_cell(
            "import sys\n"
            'sys.path.insert(0, "..")  # make src/ importable\n\n'
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            'plt.rcParams["figure.dpi"] = 120\n'
            'plt.rcParams["axes.spines.top"] = False\n'
            'plt.rcParams["axes.spines.right"] = False\n'
            'sns.set_palette("husl")\n'
            'print("Imports OK")'
        ),

        nbf.v4.new_markdown_cell("## 1. Load and Inspect the Data"),

        nbf.v4.new_code_cell(
            'df = pd.read_csv("../data/raw/credit_risk/credit_risk.csv")\n'
            'print(f"Shape: {df.shape}")\n'
            'print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")\n'
            "df.head()"
        ),

        nbf.v4.new_code_cell("df.info()"),

        nbf.v4.new_code_cell(
            "# Statistical summary of numeric features\n"
            "df.describe().round(2)"
        ),

        nbf.v4.new_markdown_cell(
            "## 2. Missing Values\n\n"
            "Missing data handling strategy depends on:\n"
            "- **<5%** missing → impute (median for skewed data, mean for symmetric)\n"
            "- **5-40%** missing → impute with caution, add a binary 'was_missing' indicator feature\n"
            "- **>40%** missing → consider dropping the column entirely"
        ),

        nbf.v4.new_code_cell(
            "missing = df.isnull().sum()\n"
            "missing_pct = (missing / len(df) * 100).round(2)\n"
            'missing_df = pd.DataFrame({"Count": missing, "Percent": missing_pct})\n'
            'missing_df = missing_df[missing_df["Count"] > 0]\n\n'
            "if len(missing_df) == 0:\n"
            '    print("No missing values — this dataset is complete!")\n'
            "else:\n"
            "    display(missing_df)\n"
            "    # Visualize\n"
            "    plt.figure(figsize=(10, 4))\n"
            '    missing_df["Percent"].plot(kind="bar")\n'
            '    plt.title("Missing Values by Column (%)")\n'
            '    plt.ylabel("Missing %")\n'
            "    plt.tight_layout()\n"
            "    plt.show()"
        ),

        nbf.v4.new_markdown_cell(
            "## 3. Target Variable Distribution\n\n"
            "**Class imbalance** is the most important characteristic of this dataset for modeling.\n"
            "With a ~22% positive class, a naive 'always predict No Default' model gets ~78% accuracy.\n"
            "This is why we use: Precision, Recall, F1, ROC-AUC — not just accuracy."
        ),

        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
            'counts = df["default"].value_counts()\n'
            "labels = [\"No Default (0)\", \"Default (1)\"]\n"
            "colors = [\"steelblue\", \"tomato\"]\n\n"
            "axes[0].bar(labels, counts.values, color=colors, alpha=0.85, edgecolor='white')\n"
            'axes[0].set_title("Target Class Distribution", fontsize=13, fontweight="bold")\n'
            'axes[0].set_ylabel("Count")\n'
            "for i, v in enumerate(counts.values):\n"
            "    pct = v / len(df) * 100\n"
            "    axes[0].text(i, v + 100, f'{v:,}\\n({pct:.1f}%)', ha='center', fontsize=10)\n\n"
            "axes[1].pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',\n"
            "            startangle=90, wedgeprops={'edgecolor': 'white'})\n"
            'axes[1].set_title("Class Proportions", fontsize=13, fontweight="bold")\n\n'
            "plt.tight_layout()\n"
            "plt.savefig('../models/credit_risk/eda_target_distribution.png', bbox_inches='tight')\n"
            "plt.show()\n"
            "print(f'Class imbalance ratio: {counts[0]/counts[1]:.1f}:1 (negative:positive)')\n"
            "print(f'Positive class rate: {counts[1]/len(df)*100:.1f}%')"
        ),

        nbf.v4.new_markdown_cell(
            "## 4. Demographic Feature Analysis\n\n"
            "We compare each feature's distribution between defaulters and non-defaulters.\n"
            "A feature where the distributions differ significantly is likely predictive."
        ),

        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n\n"
            "# Age\n"
            "for label in [0, 1]:\n"
            "    color = 'steelblue' if label == 0 else 'tomato'\n"
            "    lbl = 'No Default' if label == 0 else 'Default'\n"
            '    subset = df[df["default"] == label]["AGE"]\n'
            "    axes[0,0].hist(subset, bins=30, alpha=0.6, color=color,\n"
            "                  label=f'{lbl} (n={len(subset):,})', density=True, edgecolor='white')\n"
            'axes[0,0].set_title("Age Distribution by Default Status")\n'
            'axes[0,0].set_xlabel("Age")\n'
            'axes[0,0].set_ylabel("Density")\n'
            "axes[0,0].legend()\n\n"
            "# Credit Limit\n"
            "for label in [0, 1]:\n"
            "    color = 'steelblue' if label == 0 else 'tomato'\n"
            "    lbl = 'No Default' if label == 0 else 'Default'\n"
            '    subset = df[df["default"] == label]["LIMIT_BAL"]\n'
            "    axes[0,1].hist(subset, bins=30, alpha=0.6, color=color,\n"
            "                  density=True, edgecolor='white', label=lbl)\n"
            'axes[0,1].set_title("Credit Limit Distribution by Default Status")\n'
            'axes[0,1].set_xlabel("Credit Limit (NTD)")\n'
            'axes[0,1].set_ylabel("Density")\n'
            "axes[0,1].legend()\n\n"
            "# Education vs default rate\n"
            'edu_default = df.groupby(["EDUCATION", "default"]).size().unstack(fill_value=0)\n'
            "edu_pct = edu_default.div(edu_default.sum(axis=1), axis=0) * 100\n"
            "edu_labels = {1: 'Grad School', 2: 'University', 3: 'High School', 4: 'Other'}\n"
            "edu_pct.index = [edu_labels.get(i, str(i)) for i in edu_pct.index]\n"
            "edu_pct[1].plot(kind='bar', ax=axes[1,0], color='tomato', alpha=0.85, edgecolor='white')\n"
            'axes[1,0].set_title("Default Rate by Education Level")\n'
            'axes[1,0].set_ylabel("Default Rate (%)")\n'
            "axes[1,0].tick_params(axis='x', rotation=30)\n\n"
            "# Marriage vs default rate\n"
            'mar_default = df.groupby(["MARRIAGE", "default"]).size().unstack(fill_value=0)\n'
            "mar_pct = mar_default.div(mar_default.sum(axis=1), axis=0) * 100\n"
            "mar_labels = {1: 'Married', 2: 'Single', 3: 'Other'}\n"
            "mar_pct.index = [mar_labels.get(i, str(i)) for i in mar_pct.index]\n"
            "mar_pct[1].plot(kind='bar', ax=axes[1,1], color='tomato', alpha=0.85, edgecolor='white')\n"
            'axes[1,1].set_title("Default Rate by Marital Status")\n'
            'axes[1,1].set_ylabel("Default Rate (%)")\n'
            "axes[1,1].tick_params(axis='x', rotation=0)\n\n"
            "plt.tight_layout()\n"
            "plt.savefig('../models/credit_risk/eda_demographics.png', bbox_inches='tight')\n"
            "plt.show()"
        ),

        nbf.v4.new_markdown_cell(
            "## 5. Repayment History — The Strongest Predictor\n\n"
            "PAY_0 (most recent month's repayment status) is almost always the strongest\n"
            "predictor in credit models. Positive values = months delayed; -1 = paid on time.\n"
            "This confirms the intuition: **past behavior predicts future behavior.**"
        ),

        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
            "# PAY_0 vs default rate\n"
            'pay_default = df.groupby(["PAY_0", "default"]).size().unstack(fill_value=0)\n'
            "pay_pct = pay_default.div(pay_default.sum(axis=1), axis=0) * 100\n"
            "pay_pct[1].plot(kind='bar', ax=axes[0], color='tomato', alpha=0.85, edgecolor='white')\n"
            'axes[0].set_title("Default Rate by PAY_0 (Most Recent Repayment Status)")\n'
            'axes[0].set_xlabel("PAY_0 value (-2=no use, -1=on time, 1-9=months late)")\n'
            'axes[0].set_ylabel("Default Rate (%)")\n'
            "axes[0].tick_params(axis='x', rotation=0)\n\n"
            "# Correlation of all PAY columns with target\n"
            'pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]\n'
            'correlations = [df[col].corr(df["default"]) for col in pay_cols]\n'
            "bar_colors = ['tomato' if c > 0 else 'steelblue' for c in correlations]\n"
            "bars = axes[1].bar(pay_cols, correlations, color=bar_colors, alpha=0.85, edgecolor='white')\n"
            'axes[1].set_title("Pearson Correlation: PAY_* vs Default")\n'
            'axes[1].set_ylabel("Correlation with default")\n'
            "axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')\n"
            "for bar, val in zip(bars, correlations):\n"
            "    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,\n"
            "                 f'{val:.3f}', ha='center', fontsize=9)\n\n"
            "plt.tight_layout()\n"
            "plt.savefig('../models/credit_risk/eda_repayment_history.png', bbox_inches='tight')\n"
            "plt.show()"
        ),

        nbf.v4.new_markdown_cell(
            "## 6. Bill Amounts and Payments\n\n"
            "BILL_AMT vs PAY_AMT ratios reveal how much of their debt borrowers are actually paying off."
        ),

        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
            "# BILL_AMT1 distribution (log scale)\n"
            "for label in [0, 1]:\n"
            "    color = 'steelblue' if label == 0 else 'tomato'\n"
            "    lbl = 'No Default' if label == 0 else 'Default'\n"
            '    amt = df[df["default"] == label]["BILL_AMT1"]\n'
            "    amt_positive = amt[amt > 0]\n"
            "    axes[0].hist(np.log1p(amt_positive), bins=40, alpha=0.6, color=color,\n"
            "                 label=lbl, density=True, edgecolor='white')\n"
            'axes[0].set_title("Bill Amount Sep 2005 (log scale)")\n'
            'axes[0].set_xlabel("log(1 + BILL_AMT1)")\n'
            'axes[0].set_ylabel("Density")\n'
            "axes[0].legend()\n\n"
            "# Payment ratio\n"
            "df_tmp = df.copy()\n"
            "total_bill = df_tmp[['BILL_AMT1','BILL_AMT2','BILL_AMT3']].sum(axis=1)\n"
            "total_pay  = df_tmp[['PAY_AMT1','PAY_AMT2','PAY_AMT3']].sum(axis=1)\n"
            "df_tmp['payment_ratio'] = total_pay / (total_bill + 1)  # +1 avoids division by zero\n"
            "df_tmp['payment_ratio'] = df_tmp['payment_ratio'].clip(0, 2)  # clip extreme values\n\n"
            "for label in [0, 1]:\n"
            "    color = 'steelblue' if label == 0 else 'tomato'\n"
            "    lbl = 'No Default' if label == 0 else 'Default'\n"
            "    subset = df_tmp[df_tmp['default'] == label]['payment_ratio']\n"
            "    axes[1].hist(subset, bins=40, alpha=0.6, color=color, density=True,\n"
            "                 edgecolor='white', label=lbl)\n"
            "axes[1].set_title('Payment-to-Bill Ratio Distribution')\n"
            "axes[1].set_xlabel('Total Payments / Total Bills (clipped at 2)')\n"
            "axes[1].set_ylabel('Density')\n"
            "axes[1].legend()\n\n"
            "plt.tight_layout()\n"
            "plt.savefig('../models/credit_risk/eda_payments.png', bbox_inches='tight')\n"
            "plt.show()"
        ),

        nbf.v4.new_markdown_cell(
            "## 7. Correlation Matrix\n\n"
            "High inter-feature correlation (multicollinearity) matters differently by model type:\n"
            "- **Logistic Regression**: very sensitive — correlated features cause unstable coefficients\n"
            "- **XGBoost**: handles it well — trees split on one feature at a time\n"
            "- **SHAP**: multicollinearity can split importance across correlated features"
        ),

        nbf.v4.new_code_cell(
            "feature_cols = [c for c in df.columns if c not in ['ID', 'default']]\n"
            "corr_matrix = df[feature_cols + ['default']].corr()\n\n"
            "plt.figure(figsize=(18, 14))\n"
            "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n"
            "sns.heatmap(\n"
            "    corr_matrix, mask=mask, annot=True, fmt='.2f',\n"
            "    cmap='RdBu_r', center=0, vmin=-1, vmax=1,\n"
            "    square=True, linewidths=0.4, cbar_kws={'shrink': 0.8},\n"
            "    annot_kws={'size': 7}\n"
            ")\n"
            "plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.savefig('../models/credit_risk/eda_correlation_matrix.png', bbox_inches='tight')\n"
            "plt.show()\n\n"
            "# Top features correlated with target\n"
            "target_corr = corr_matrix['default'].drop('default').abs().sort_values(ascending=False)\n"
            "print('Top 10 features correlated with default (by absolute Pearson r):')\n"
            "print(target_corr.head(10).round(4).to_string())"
        ),

        nbf.v4.new_markdown_cell(
            "## 8. EDA Summary — What We Learned\n\n"
            "| Finding | Implication |\n"
            "|---|---|\n"
            "| No missing values | No imputation needed; pipeline adds it defensively |\n"
            "| 22% positive class | Use `scale_pos_weight` in XGBoost; tune decision threshold |\n"
            "| PAY_0–PAY_6 most predictive | Recent repayment history dominates |\n"
            "| High correlation in BILL_AMT group | XGBoost handles it; SHAP will split importance |\n"
            "| Higher LIMIT_BAL → lower default | Wealthier borrowers default less |\n\n"
            "### Feature Engineering Ideas for Phase 2\n"
            "```python\n"
            "# Aggregate repayment trend\n"
            "df['pay_delay_mean']    = df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].mean(axis=1)\n"
            "df['pay_delay_max']     = df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].max(axis=1)\n\n"
            "# Financial ratios\n"
            "df['total_bill']    = df[['BILL_AMT1','BILL_AMT2','BILL_AMT3']].sum(axis=1)\n"
            "df['total_payment'] = df[['PAY_AMT1','PAY_AMT2','PAY_AMT3']].sum(axis=1)\n"
            "df['payment_ratio'] = df['total_payment'] / (df['total_bill'] + 1)\n"
            "```"
        ),
    ]

    nb.cells = cells
    nbf.write(nb, str(NB_DIR / "01_eda_credit_risk.ipynb"))
    print("Created: notebooks/01_eda_credit_risk.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 2: Network Intrusion EDA
# ─────────────────────────────────────────────────────────────────────────────

def make_network_eda():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

    column_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label","difficulty_level"
    ]

    col_names_str = str(column_names)

    cells = [
        nbf.v4.new_markdown_cell(
            "# Network Intrusion Detection — Exploratory Data Analysis\n\n"
            "**Dataset**: NSL-KDD (improved version of the classic KDD Cup 1999 dataset)  \n"
            "**Train records**: 125,973 | **Test records**: 22,544 | **Features**: 41\n\n"
            "**Target**: Binary — `normal` (0) vs. any attack type (1)\n\n"
            "## What Makes NSL-KDD Different from Credit Risk?\n"
            "- **Mixed feature types**: continuous + binary + categorical (protocol, service, flag)\n"
            "- **Multi-class labels** that we binarize (neptune, smurf, back → all 'attack')\n"
            "- **Pre-split train/test**: The dataset provides official splits — we respect them\n"
            "- **Network domain features**: traffic statistics, connection counts, error rates"
        ),

        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.insert(0, '..')\n\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "plt.rcParams['figure.dpi'] = 120\n"
            "plt.rcParams['axes.spines.top'] = False\n"
            "plt.rcParams['axes.spines.right'] = False\n"
            "sns.set_palette('husl')\n"
            "print('Imports OK')"
        ),

        nbf.v4.new_markdown_cell("## 1. Load the Data\n\nNSL-KDD has no header row — we supply column names manually."),

        nbf.v4.new_code_cell(
            "column_names = " + col_names_str + "\n\n"
            "train_df = pd.read_csv('../data/raw/network_intrusion/KDDTrain+.txt',\n"
            "                       header=None, names=column_names)\n"
            "test_df  = pd.read_csv('../data/raw/network_intrusion/KDDTest+.txt',\n"
            "                       header=None, names=column_names)\n\n"
            "# Drop the difficulty_level meta column\n"
            "train_df = train_df.drop(columns=['difficulty_level'])\n"
            "test_df  = test_df.drop(columns=['difficulty_level'])\n\n"
            "print(f'Train shape: {train_df.shape}')\n"
            "print(f'Test shape:  {test_df.shape}')\n"
            "train_df.head()"
        ),

        nbf.v4.new_markdown_cell(
            "## 2. Label Analysis\n\n"
            "The raw labels are attack type names (neptune, smurf, etc.).\n"
            "We binarize them: `normal` → 0, everything else → 1 (attack).\n"
            "This makes it a binary classification problem."
        ),

        nbf.v4.new_code_cell(
            "# Raw label distribution\n"
            "print('=== Training Set Raw Labels ===')\n"
            "print(train_df['label'].value_counts().to_string())\n\n"
            "# Binarize\n"
            "train_df['binary_label'] = (train_df['label'] != 'normal').astype(int)\n"
            "test_df['binary_label']  = (test_df['label']  != 'normal').astype(int)\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
            "# Train\n"
            "counts_train = train_df['binary_label'].value_counts()\n"
            "axes[0].bar(['Normal (0)', 'Attack (1)'], counts_train.values,\n"
            "            color=['steelblue', 'tomato'], alpha=0.85, edgecolor='white')\n"
            "axes[0].set_title('Train Set Class Distribution')\n"
            "axes[0].set_ylabel('Count')\n"
            "for i, v in enumerate(counts_train.values):\n"
            "    pct = v / len(train_df) * 100\n"
            "    axes[0].text(i, v + 500, f'{v:,}\\n({pct:.1f}%)', ha='center', fontsize=10)\n\n"
            "# Test\n"
            "counts_test = test_df['binary_label'].value_counts()\n"
            "axes[1].bar(['Normal (0)', 'Attack (1)'], counts_test.values,\n"
            "            color=['steelblue', 'tomato'], alpha=0.85, edgecolor='white')\n"
            "axes[1].set_title('Test Set Class Distribution')\n"
            "axes[1].set_ylabel('Count')\n"
            "for i, v in enumerate(counts_test.values):\n"
            "    pct = v / len(test_df) * 100\n"
            "    axes[1].text(i, v + 200, f'{v:,}\\n({pct:.1f}%)', ha='center', fontsize=10)\n\n"
            "plt.tight_layout()\n"
            "plt.savefig('../models/network_intrusion/eda_class_distribution.png', bbox_inches='tight')\n"
            "plt.show()"
        ),

        nbf.v4.new_markdown_cell(
            "## 3. Categorical Features\n\n"
            "Three categorical features need special handling:\n"
            "- `protocol_type`: tcp, udp, icmp — network protocol\n"
            "- `service`: http, ftp, smtp, etc. — network service\n"
            "- `flag`: SF, S0, REJ, etc. — connection status\n\n"
            "We'll use **OneHotEncoder** for these (not label encoding) to avoid\n"
            "implying any ordinal relationship between categories."
        ),

        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n\n"
            "for ax, col in zip(axes, ['protocol_type', 'service', 'flag']):\n"
            "    vc = train_df[col].value_counts()\n"
            "    # Show only top 15 for service (has many values)\n"
            "    if len(vc) > 15:\n"
            "        other_count = vc[15:].sum()\n"
            "        vc = pd.concat([vc[:15], pd.Series({'other': other_count})])\n"
            "    vc.plot(kind='bar', ax=ax, color='steelblue', alpha=0.85, edgecolor='white')\n"
            "    ax.set_title(f'{col} distribution')\n"
            "    ax.set_ylabel('Count')\n"
            "    ax.tick_params(axis='x', rotation=45)\n\n"
            "plt.tight_layout()\n"
            "plt.savefig('../models/network_intrusion/eda_categoricals.png', bbox_inches='tight')\n"
            "plt.show()\n\n"
            "# Attack rate by protocol\n"
            "print('Attack rate by protocol_type:')\n"
            "print(train_df.groupby('protocol_type')['binary_label'].mean().round(3).to_string())"
        ),

        nbf.v4.new_markdown_cell(
            "## 4. Numeric Feature Distributions\n\n"
            "Many network features (src_bytes, dst_bytes, count) are highly skewed —\n"
            "normal traffic often has near-zero values while attacks have extreme spikes."
        ),

        nbf.v4.new_code_cell(
            "key_numeric = ['duration', 'src_bytes', 'dst_bytes', 'count',\n"
            "               'srv_count', 'serror_rate', 'rerror_rate', 'same_srv_rate']\n\n"
            "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n"
            "axes = axes.flatten()\n\n"
            "for ax, col in zip(axes, key_numeric):\n"
            "    for label in [0, 1]:\n"
            "        color = 'steelblue' if label == 0 else 'tomato'\n"
            "        lbl = 'Normal' if label == 0 else 'Attack'\n"
            "        vals = train_df[train_df['binary_label'] == label][col]\n"
            "        ax.hist(np.log1p(vals), bins=30, alpha=0.6, color=color,\n"
            "                density=True, edgecolor='white', label=lbl)\n"
            "    ax.set_title(f'{col} (log scale)')\n"
            "    ax.set_xlabel('log(1+value)')\n"
            "    ax.legend(fontsize=7)\n\n"
            "plt.suptitle('Key Numeric Features: Normal vs Attack Traffic', fontsize=13, y=1.01)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../models/network_intrusion/eda_numeric_features.png', bbox_inches='tight')\n"
            "plt.show()"
        ),

        nbf.v4.new_markdown_cell(
            "## 5. Feature Correlation with Target\n\n"
            "Point-biserial correlation between each numeric feature and the binary label.\n"
            "This gives us a quick signal of which features are most discriminative."
        ),

        nbf.v4.new_code_cell(
            "numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()\n"
            "numeric_cols = [c for c in numeric_cols if c not in ['binary_label']]\n\n"
            "corr_with_target = (\n"
            "    train_df[numeric_cols + ['binary_label']]\n"
            "    .corr()['binary_label']\n"
            "    .drop('binary_label')\n"
            "    .abs()\n"
            "    .sort_values(ascending=False)\n"
            ")\n\n"
            "plt.figure(figsize=(12, 7))\n"
            "top20 = corr_with_target.head(20)\n"
            "colors = ['tomato' if v > 0.3 else 'steelblue' for v in top20.values]\n"
            "plt.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1], alpha=0.85, edgecolor='white')\n"
            "plt.title('Top 20 Features by Absolute Correlation with Attack Label', fontsize=12)\n"
            "plt.xlabel('|Pearson Correlation|')\n"
            "plt.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='0.3 threshold')\n"
            "plt.legend()\n"
            "plt.tight_layout()\n"
            "plt.savefig('../models/network_intrusion/eda_feature_correlations.png', bbox_inches='tight')\n"
            "plt.show()\n\n"
            "print('Top 15 features by correlation with attack label:')\n"
            "print(corr_with_target.head(15).round(4).to_string())"
        ),

        nbf.v4.new_markdown_cell(
            "## 6. Train/Test Distribution Check\n\n"
            "NSL-KDD's test set is intentionally harder than the train set — it contains\n"
            "attack types not seen during training. This is realistic but means our test\n"
            "metrics may be lower than train metrics even without overfitting."
        ),

        nbf.v4.new_code_cell(
            "# Compare attack type distributions between train and test\n"
            "train_attack_types = train_df[train_df['binary_label']==1]['label'].value_counts(normalize=True)\n"
            "test_attack_types  = test_df[test_df['binary_label']==1]['label'].value_counts(normalize=True)\n\n"
            "comparison = pd.DataFrame({\n"
            "    'Train': train_attack_types,\n"
            "    'Test': test_attack_types\n"
            "}).fillna(0).round(4)\n\n"
            "comparison.plot(kind='bar', figsize=(14, 5), color=['steelblue', 'tomato'],\n"
            "               alpha=0.85, edgecolor='white')\n"
            "plt.title('Attack Type Distribution: Train vs Test')\n"
            "plt.ylabel('Proportion')\n"
            "plt.xlabel('Attack Type')\n"
            "plt.xticks(rotation=45, ha='right')\n"
            "plt.tight_layout()\n"
            "plt.savefig('../models/network_intrusion/eda_train_test_comparison.png', bbox_inches='tight')\n"
            "plt.show()\n\n"
            "# Flag attack types in test but not in train\n"
            "train_types = set(train_df['label'].unique())\n"
            "test_types  = set(test_df['label'].unique())\n"
            "novel_types = test_types - train_types\n"
            "if novel_types:\n"
            "    print(f'Attack types in TEST but NOT in TRAIN: {novel_types}')\n"
            "    print('These unseen attack types make this a realistic but harder generalization test.')"
        ),

        nbf.v4.new_markdown_cell(
            "## 7. EDA Summary — What We Learned\n\n"
            "| Finding | Implication |\n"
            "|---|---|\n"
            "| 3 categorical features (protocol, service, flag) | OneHotEncoder in pipeline |\n"
            "| Highly skewed numeric features | StandardScaler handles this |\n"
            "| serror_rate, rerror_rate highly correlated with attack | Strong features for model |\n"
            "| Test set has novel attack types | Realistic generalization challenge |\n"
            "| ~52% attacks in train set | Near-balanced — less imbalance than credit risk |\n\n"
            "### Key Features to Watch (from correlation analysis)\n"
            "1. `serror_rate` / `dst_host_serror_rate` — SYN flood attacks cause high SYN error rates\n"
            "2. `logged_in` — Attacks often happen without a successful login\n"
            "3. `dst_host_srv_count` — DoS attacks generate many connections to the same service\n"
            "4. `flag` (SF=normal connection vs S0/REJ=failed) — huge discriminative power after encoding"
        ),
    ]

    nb.cells = cells
    nbf.write(nb, str(NB_DIR / "02_eda_network_intrusion.ipynb"))
    print("Created: notebooks/02_eda_network_intrusion.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 3: Model Experiments (post-training comparison)
# ─────────────────────────────────────────────────────────────────────────────

def make_model_experiments():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

    cells = [
        nbf.v4.new_markdown_cell(
            "# Model Experiments — Evaluation & Comparison\n\n"
            "Run this notebook AFTER training is complete (`python -m src.models.train --domain all`).\n\n"
            "This notebook:\n"
            "1. Loads saved evaluation reports from both domains\n"
            "2. Compares baseline vs tuned XGBoost\n"
            "3. Displays confusion matrices and ROC curves\n"
            "4. Shows SHAP explainability plots"
        ),

        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.insert(0, '..')\n\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from pathlib import Path\n\n"
            "plt.rcParams['figure.dpi'] = 120\n"
            "plt.rcParams['axes.spines.top'] = False\n"
            "plt.rcParams['axes.spines.right'] = False\n\n"
            "ROOT = Path('..')\n"
            "print('Imports OK')"
        ),

        nbf.v4.new_markdown_cell("## 1. Load Evaluation Reports"),

        nbf.v4.new_code_cell(
            "def load_report(domain):\n"
            "    path = ROOT / 'models' / domain / 'evaluation_report.json'\n"
            "    if not path.exists():\n"
            "        print(f'No report for {domain} — run training first')\n"
            "        return None\n"
            "    with open(path) as f:\n"
            "        return json.load(f)\n\n"
            "cr_report = load_report('credit_risk')\n"
            "ni_report = load_report('network_intrusion')\n\n"
            "if cr_report and ni_report:\n"
            "    print('Both reports loaded successfully')\n"
            "    print('\\nCredit Risk metrics:', cr_report['scalar_metrics'])\n"
            "    print('\\nNetwork Intrusion metrics:', ni_report['scalar_metrics'])"
        ),

        nbf.v4.new_markdown_cell("## 2. Metrics Comparison Table"),

        nbf.v4.new_code_cell(
            "if cr_report and ni_report:\n"
            "    metrics_df = pd.DataFrame({\n"
            "        'Credit Risk': cr_report['scalar_metrics'],\n"
            "        'Network Intrusion': ni_report['scalar_metrics']\n"
            "    }).T[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]\n"
            "    metrics_df.round(4)"
        ),

        nbf.v4.new_markdown_cell(
            "## 3. Confusion Matrices\n\n"
            "The confusion matrix shows the four prediction outcomes:\n"
            "- **True Negative (TN)**: Correctly predicted safe\n"
            "- **False Positive (FP)**: Wrongly flagged as risky (false alarm)\n"
            "- **False Negative (FN)**: Missed a real risk (dangerous in finance!)\n"
            "- **True Positive (TP)**: Correctly predicted risky\n\n"
            "For risk prediction, **False Negatives are worse** — missing a default costs more than a false alarm."
        ),

        nbf.v4.new_code_cell(
            "if cr_report and ni_report:\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n\n"
            "    for ax, report, title in zip(\n"
            "        axes,\n"
            "        [cr_report, ni_report],\n"
            "        ['Credit Risk', 'Network Intrusion']\n"
            "    ):\n"
            "        cm = np.array(report['confusion_matrix'])\n"
            "        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n"
            "                    xticklabels=['Pred Neg', 'Pred Pos'],\n"
            "                    yticklabels=['Actual Neg', 'Actual Pos'],\n"
            "                    cbar=False)\n"
            "        tn, fp, fn, tp = cm.ravel()\n"
            "        ax.set_title(f'{title}\\nROC-AUC={report[\"scalar_metrics\"][\"roc_auc\"]:.4f}')\n\n"
            "    plt.tight_layout()\n"
            "    plt.savefig('../models/confusion_matrices.png', bbox_inches='tight')\n"
            "    plt.show()"
        ),

        nbf.v4.new_markdown_cell("## 4. SHAP Explainability Plots"),

        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n\n"
            "for domain in ['credit_risk', 'network_intrusion']:\n"
            "    shap_path = ROOT / 'models' / domain / 'shap_summary.png'\n"
            "    if shap_path.exists():\n"
            "        print(f'\\n=== {domain.replace(\"_\", \" \").title()} SHAP Summary ===')\n"
            "        display(Image(str(shap_path), width=800))\n"
            "    else:\n"
            "        print(f'SHAP plots not found for {domain} — run training first')"
        ),
    ]

    nb.cells = cells
    nbf.write(nb, str(NB_DIR / "03_model_experiments.ipynb"))
    print("Created: notebooks/03_model_experiments.ipynb")


if __name__ == "__main__":
    make_credit_eda()
    make_network_eda()
    make_model_experiments()
    print("\nAll notebooks created successfully in notebooks/")
