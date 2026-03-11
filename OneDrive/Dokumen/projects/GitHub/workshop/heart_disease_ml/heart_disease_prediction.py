# ══════════════════════════════════════════════════════════════════════════════
#  Heart Disease Prediction — Core ML Pipeline
#  Pragyan AI Hackathon | Classification Project
#  Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  HEART DISEASE PREDICTION — ML PIPELINE")
print("=" * 60)

# Try multiple reliable sources for the heart disease dataset
URLS = [
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
    "https://raw.githubusercontent.com/sureshHARDIYA/introML/master/Heart_Disease/data.csv",
]

# Column names for UCI dataset (if needed)
UCI_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = None
for url in URLS:
    try:
        print(f"\nAttempting to load dataset from: {url.split('/')[-2]}/{url.split('/')[-1]}")
        df = pd.read_csv(url)
        print(f"Dataset loaded successfully | Shape: {df.shape}")
        break
    except Exception as e:
        print(f"Failed: {str(e)[:60]}")
        continue

# Try UCI dataset with proper column names if other sources failed
if df is None:
    try:
        print(f"\nAttempting to load from UCI repository...")
        df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            names=UCI_COLUMNS
        )
        # Convert target: 0 = no disease, 1-4 = disease present
        df['target'] = (df['target'] > 0).astype(int)
        # Replace '?' with NaN and handle missing values
        df = df.replace('?', np.nan)
        df = df.dropna()
        # Convert columns to appropriate types
        for col in df.columns:
            if col != 'target':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        print(f"Dataset loaded from UCI repository | Shape: {df.shape}")
    except Exception as e:
        print(f"Failed: {str(e)[:60]}")

# Final fallback: try local file
if df is None:
    try:
        df = pd.read_csv("heart.csv")
        print(f"\nDataset loaded from local file | Shape: {df.shape}")
    except FileNotFoundError:
        print("\nERROR: Could not load dataset from any source.")
        print("Please download heart.csv from:")
        print("https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
        print("and place it in the project directory.")
        exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. EDA ──────────────────────────────────────────────")
print(df.head())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nTarget Distribution:\n", df["target"].value_counts())
print("\nStatistical Summary:\n", df.describe())

# Plot 1: Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["target"].value_counts().plot(kind="bar", ax=axes[0], color=["#2d6a4f","#e63946"],
                                  edgecolor="black")
axes[0].set_title("Target Distribution (0=No Disease, 1=Disease)")
axes[0].set_xlabel("Target"); axes[0].set_ylabel("Count")
axes[0].tick_params(rotation=0)

# Plot 2: Age Distribution by Target
for t, color, lbl in [(0,"#2d6a4f","No Disease"), (1,"#e63946","Disease")]:
    axes[1].hist(df[df["target"]==t]["age"], bins=15, alpha=0.7, color=color, label=lbl)
axes[1].set_title("Age Distribution by Target")
axes[1].set_xlabel("Age"); axes[1].set_ylabel("Count")
axes[1].legend()
plt.tight_layout()
plt.savefig("eda_distribution.png", dpi=150)
plt.show()
print("Saved: eda_distribution.png")

# Plot 3: Correlation Heatmap
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", linewidths=0.3, annot_kws={"size": 8})
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.show()
print("Saved: eda_correlation.png")

# Plot 4: Chest Pain vs Disease
plt.figure(figsize=(7, 4))
cp_counts = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
cp_counts.plot(kind="bar", color=["#2d6a4f","#e63946"], edgecolor="black")
plt.title("Chest Pain Type vs Disease"); plt.xlabel("Chest Pain Type")
plt.ylabel("Count"); plt.legend(["No Disease","Disease"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_chestpain.png", dpi=150)
plt.show()
print("Saved: eda_chestpain.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. PREPROCESSING ────────────────────────────────────")

FEATURES = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"]
TARGET   = "target"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print("Features standardized with StandardScaler")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. MODEL TRAINING ───────────────────────────────────")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)
    cv    = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy").mean()

    results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob,
                     "accuracy": acc, "roc_auc": auc, "cv_accuracy": cv}
    print(f"\n  {name}")
    print(f"    Accuracy     : {acc:.4f}")
    print(f"    ROC-AUC      : {auc:.4f}")
    print(f"    CV Accuracy  : {cv:.4f} (5-fold)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. EVALUATION ───────────────────────────────────────")

# Best model
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best      = results[best_name]
print(f"\nBest Model: {best_name} (ROC-AUC = {best['roc_auc']:.4f})")

print(f"\nClassification Report — {best_name}:\n")
print(classification_report(y_test, best["y_pred"],
                             target_names=["No Disease", "Disease"]))

# Plot 5: Confusion Matrices (all models)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm   = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease","Disease"])
    disp.plot(ax=ax, colorbar=False, cmap="RdYlGn")
    ax.set_title(f"{name}\nAcc={res['accuracy']:.2%} | AUC={res['roc_auc']:.3f}")
plt.tight_layout()
plt.savefig("eval_confusion_matrices.png", dpi=150)
plt.show()
print("Saved: eval_confusion_matrices.png")

# Plot 6: ROC Curves
plt.figure(figsize=(7, 5))
colors = ["#4cc9f0", "#ffd60a", "#e63946"]
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC={res['roc_auc']:.3f})")
plt.plot([0,1],[0,1], "k--", lw=1, label="Random")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — All Models"); plt.legend()
plt.tight_layout()
plt.savefig("eval_roc_curves.png", dpi=150)
plt.show()
print("Saved: eval_roc_curves.png")

# Plot 7: Feature Importance (Random Forest)
rf_model  = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
colors = ["#e63946" if v > feat_imp.median() else "#4cc9f0" for v in feat_imp.values]
feat_imp.plot(kind="barh", color=colors, edgecolor="none")
plt.title("Feature Importances — Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("eval_feature_importance.png", dpi=150)
plt.show()
print("Saved: eval_feature_importance.png")

# Plot 8: Model Comparison Bar Chart
plt.figure(figsize=(7, 4))
names = list(results.keys())
accs  = [results[n]["accuracy"] for n in names]
aucs  = [results[n]["roc_auc"]  for n in names]
x     = np.arange(len(names))
width = 0.35
plt.bar(x - width/2, accs, width, label="Accuracy",  color="#4cc9f0", edgecolor="black")
plt.bar(x + width/2, aucs, width, label="ROC-AUC",   color="#e63946", edgecolor="black")
plt.xticks(x, names, rotation=10)
plt.ylabel("Score"); plt.title("Model Comparison")
plt.ylim(0.7, 1.0); plt.legend()
plt.tight_layout()
plt.savefig("eval_model_comparison.png", dpi=150)
plt.show()
print("Saved: eval_model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAMPLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 6. SAMPLE PREDICTION ────────────────────────────────")

# Example patient: age=55, male, chest pain type=2, bp=140, chol=250...
sample = pd.DataFrame([[55, 1, 2, 140, 250, 0, 1, 160, 0, 1.2, 1, 0, 2]],
                      columns=FEATURES)
sample_scaled = scaler.transform(sample)

best_model = results[best_name]["model"]
pred       = best_model.predict(sample_scaled)[0]
prob       = best_model.predict_proba(sample_scaled)[0]

print(f"\nPatient Input: {sample.to_dict('records')[0]}")
print(f"\nModel Used   : {best_name}")
print(f"Prediction   : {'HEART DISEASE DETECTED' if pred == 1 else 'No Disease'}")
print(f"Probability  : Disease={prob[1]:.2%} | No Disease={prob[0]:.2%}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE — All plots saved as PNG files")
print("=" * 60)
