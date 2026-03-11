import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stButton>button {
        background: linear-gradient(135deg, #e63946, #c1121f);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: bold; font-size: 1rem;
    }
    .stButton>button:hover { opacity: 0.85; transform: scale(1.02); }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-left: 4px solid #e63946;
        border-radius: 10px; padding: 1rem;
        margin: 0.5rem 0;
    }
    .section-title {
        font-size: 1.4rem; font-weight: bold;
        color: #e63946; margin: 1rem 0 0.5rem 0;
    }
    .predict-positive {
        background: linear-gradient(135deg, #e63946, #c1121f);
        color: white; border-radius: 10px; padding: 1.5rem;
        text-align: center; font-size: 1.3rem; font-weight: bold;
    }
    .predict-negative {
        background: linear-gradient(135deg, #2d6a4f, #1b4332);
        color: white; border-radius: 10px; padding: 1.5rem;
        text-align: center; font-size: 1.3rem; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/anatomical-heart.png", width=80)
    st.title("Heart Disease\nPredictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "EDA", "Model Training", "Predict"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Dataset: UCI Heart Disease\nvia Kaggle")

# ─── Load & cache data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Bundled dataset (same as Kaggle heart-disease-dataset)
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # Fallback: generate realistic synthetic data for demo
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            "age":      np.random.randint(29, 77, n),
            "sex":      np.random.randint(0, 2, n),
            "cp":       np.random.randint(0, 4, n),
            "trestbps": np.random.randint(94, 200, n),
            "chol":     np.random.randint(126, 564, n),
            "fbs":      np.random.randint(0, 2, n),
            "restecg":  np.random.randint(0, 3, n),
            "thalach":  np.random.randint(71, 202, n),
            "exang":    np.random.randint(0, 2, n),
            "oldpeak":  np.round(np.random.uniform(0, 6.2, n), 1),
            "slope":    np.random.randint(0, 3, n),
            "ca":       np.random.randint(0, 5, n),
            "thal":     np.random.randint(0, 4, n),
            "target":   np.random.randint(0, 2, n),
        })
    return df

@st.cache_resource
def train_models(df):
    feature_cols = ["age","sex","cp","trestbps","chol","fbs",
                    "restecg","thalach","exang","oldpeak","slope","ca","thal"]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train_s, y_train)

    return {
        "scaler": scaler,
        "lr": lr, "rf": rf,
        "X_test": X_test, "y_test": y_test,
        "X_test_s": X_test_s,
        "feature_cols": feature_cols
    }

df = load_data()
models = train_models(df)

FEATURE_LABELS = {
    "age":      "Age",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 (1=True)",
    "restecg":  "Resting ECG (0-2)",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise Induced Angina (1=Yes)",
    "oldpeak":  "ST Depression (Oldpeak)",
    "slope":    "Slope of ST Segment (0-2)",
    "ca":       "No. of Major Vessels (0-4)",
    "thal":     "Thalassemia (0-3)",
}

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Heart Disease Prediction System")
    st.markdown("**Pragyan AI Hackathon Project** — Classification using ML")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Disease Cases", int(df["target"].sum()))
    with col3:
        st.metric("Healthy Cases", int((df["target"] == 0).sum()))
    with col4:
        st.metric("Features", 13)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Problem Statement")
        st.info("""
Build a model to **predict whether a patient has heart disease**
based on 13 clinical measurements such as age, cholesterol,
blood pressure, and chest pain type.

**Target:** `target` — 0 = No Disease, 1 = Disease
        """)
        st.markdown("### Models Used")
        st.success("Logistic Regression\n\nRandom Forest Classifier")

    with col_r:
        st.markdown("### Feature Description")
        feat_df = pd.DataFrame(FEATURE_LABELS.items(), columns=["Feature", "Description"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Raw Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: EDA
# ════════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    # Target distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f1117")
        counts = df["target"].value_counts()
        bars = ax.bar(["No Disease (0)", "Disease (1)"], counts.values,
                      color=["#2d6a4f", "#e63946"], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", color="white", fontweight="bold")
        ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
        ax.spines[:].set_color("#444"); ax.yaxis.label.set_color("white")
        ax.set_ylabel("Count", color="white")
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### Age Distribution by Target")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f1117")
        ax.set_facecolor("#1e1e2e")
        for target, color, label in [(0,"#2d6a4f","No Disease"), (1,"#e63946","Disease")]:
            subset = df[df["target"] == target]["age"]
            ax.hist(subset, bins=15, alpha=0.7, color=color, label=label, edgecolor="white", lw=0.3)
        ax.legend(facecolor="#1e1e2e", labelcolor="white")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
        ax.set_xlabel("Age", color="white"); ax.set_ylabel("Count", color="white")
        fig.tight_layout()
        st.pyplot(fig)

    # Correlation heatmap
    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0f1117")
    ax.set_facecolor("#1e1e2e")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=ax, linewidths=0.3, linecolor="#333",
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.tick_params(colors="white", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    # Chest pain vs target
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Chest Pain Type vs Disease")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f1117")
        ax.set_facecolor("#1e1e2e")
        cp_counts = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
        cp_counts.plot(kind="bar", ax=ax, color=["#2d6a4f","#e63946"],
                       edgecolor="white", lw=0.4)
        ax.set_xlabel("Chest Pain Type", color="white")
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white", rotation=0)
        ax.legend(["No Disease","Disease"], facecolor="#1e1e2e", labelcolor="white")
        ax.spines[:].set_color("#444")
        fig.tight_layout()
        st.pyplot(fig)

    with col4:
        st.markdown("#### Max Heart Rate vs Cholesterol")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f1117")
        ax.set_facecolor("#1e1e2e")
        for target, color, label in [(0,"#2d6a4f","No Disease"), (1,"#e63946","Disease")]:
            subset = df[df["target"] == target]
            ax.scatter(subset["chol"], subset["thalach"], alpha=0.6,
                       color=color, label=label, s=20)
        ax.set_xlabel("Cholesterol", color="white")
        ax.set_ylabel("Max Heart Rate", color="white")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
        ax.legend(facecolor="#1e1e2e", labelcolor="white")
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("#### Statistical Summary")
    st.dataframe(df.describe().T.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Model Training":
    st.title("Model Training & Evaluation")
    st.markdown("---")

    scaler   = models["scaler"]
    lr       = models["lr"]
    rf       = models["rf"]
    X_test_s = models["X_test_s"]
    y_test   = models["y_test"]
    feat_cols = models["feature_cols"]

    # Metrics
    def get_metrics(model, X, y):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        report = classification_report(y, y_pred, output_dict=True)
        auc = roc_auc_score(y, y_prob)
        return y_pred, y_prob, report, auc

    lr_pred, lr_prob, lr_rep, lr_auc = get_metrics(lr, X_test_s, y_test)
    rf_pred, rf_prob, rf_rep, rf_auc = get_metrics(rf, X_test_s, y_test)

    st.markdown("### Model Performance Comparison")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("LR Accuracy",  f"{lr_rep['accuracy']:.2%}")
    with col2:
        st.metric("LR ROC-AUC",   f"{lr_auc:.3f}")
    with col3:
        st.metric("RF Accuracy",  f"{rf_rep['accuracy']:.2%}")
    with col4:
        st.metric("RF ROC-AUC",   f"{rf_auc:.3f}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrices", "ROC Curves", "Feature Importance"])

    with tab1:
        c1, c2 = st.columns(2)
        for col, model, pred, name in [
            (c1, lr, lr_pred, "Logistic Regression"),
            (c2, rf, rf_pred, "Random Forest")
        ]:
            with col:
                st.markdown(f"##### {name}")
                fig, ax = plt.subplots(figsize=(4, 3), facecolor="#0f1117")
                ax.set_facecolor("#1e1e2e")
                cm = confusion_matrix(y_test, pred)
                disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease","Disease"])
                disp.plot(ax=ax, colorbar=False, cmap="RdYlGn")
                ax.tick_params(colors="white", labelsize=8)
                ax.set_xlabel("Predicted", color="white")
                ax.set_ylabel("Actual", color="white")
                for text in ax.texts:
                    text.set_color("white"); text.set_fontsize(12)
                fig.tight_layout()
                st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f1117")
        ax.set_facecolor("#1e1e2e")
        for prob, name, color in [
            (lr_prob, f"Logistic Regression (AUC={lr_auc:.3f})", "#4cc9f0"),
            (rf_prob, f"Random Forest (AUC={rf_auc:.3f})", "#e63946"),
        ]:
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, color=color, lw=2, label=name)
        ax.plot([0,1],[0,1], "w--", lw=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", color="white")
        ax.set_ylabel("True Positive Rate", color="white")
        ax.set_title("ROC Curve Comparison", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1e1e2e", labelcolor="white")
        ax.spines[:].set_color("#444")
        fig.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.markdown("##### Random Forest Feature Importances")
        importances = rf.feature_importances_
        feat_imp = pd.Series(importances, index=feat_cols).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f1117")
        ax.set_facecolor("#1e1e2e")
        colors = ["#e63946" if v > feat_imp.median() else "#4cc9f0" for v in feat_imp.values]
        ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor="none")
        ax.set_xlabel("Importance Score", color="white")
        ax.tick_params(colors="white", labelsize=9)
        ax.spines[:].set_color("#444")
        ax.set_title("Feature Importances (Random Forest)", color="white")
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Classification Report — Random Forest")
    report_df = pd.DataFrame(rf_rep).T.drop("accuracy").round(3)
    st.dataframe(report_df.style.background_gradient(cmap="RdYlGn", subset=["precision","recall","f1-score"]),
                 use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Predict Heart Disease Risk")
    st.markdown("Enter patient details below to get an instant prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.slider("Age", 20, 80, 50)
        sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        cp       = st.selectbox("Chest Pain Type", [0,1,2,3],
                                format_func=lambda x: f"{x} - {['Typical Angina','Atypical Angina','Non-Anginal','Asymptomatic'][x]}")
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 200)

    with col2:
        fbs      = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        restecg  = st.selectbox("Resting ECG", [0,1,2],
                                format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
        thalach  = st.slider("Max Heart Rate", 60, 210, 150)
        exang    = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

    with col3:
        oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
        slope    = st.selectbox("Slope of ST Segment", [0,1,2],
                                format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
        ca       = st.selectbox("No. of Major Vessels (0-4)", [0,1,2,3,4])
        thal     = st.selectbox("Thalassemia", [0,1,2,3],
                                format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect","Unknown"][x])

    st.markdown("---")
    model_choice = st.radio("Select Model", ["Random Forest", "Logistic Regression"], horizontal=True)

    if st.button("Predict Now"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = models["scaler"].transform(input_data)

        model = models["rf"] if model_choice == "Random Forest" else models["lr"]
        pred  = model.predict(input_scaled)[0]
        prob  = model.predict_proba(input_scaled)[0]

        st.markdown("### Prediction Result")
        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class="predict-positive">
                    HIGH RISK DETECTED<br><br>
                    Heart Disease Probability: <b>{prob[1]:.1%}</b>
                </div>""", unsafe_allow_html=True)
                st.error("Please consult a cardiologist immediately.")
            else:
                st.markdown(f"""
                <div class="predict-negative">
                    LOW RISK<br><br>
                    Heart Disease Probability: <b>{prob[1]:.1%}</b>
                </div>""", unsafe_allow_html=True)
                st.success("No signs of heart disease detected. Stay healthy!")

        with col_gauge:
            fig, ax = plt.subplots(figsize=(4, 3), facecolor="#0f1117")
            ax.set_facecolor("#1e1e2e")
            bar_color = "#e63946" if pred == 1 else "#2d6a4f"
            ax.barh(["Risk"], [prob[1]], color=bar_color, height=0.4)
            ax.barh(["Risk"], [1 - prob[1]], left=prob[1], color="#333", height=0.4)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="white", linestyle="--", linewidth=1, alpha=0.5)
            ax.text(prob[1]/2, 0, f"{prob[1]:.1%}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=14)
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            ax.set_xlabel("Probability", color="white")
            ax.set_title("Risk Gauge", color="white")
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("### Input Summary")
        input_summary = pd.DataFrame({
            "Feature": list(FEATURE_LABELS.values()),
            "Value": [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                      exang, oldpeak, slope, ca, thal]
        })
        st.dataframe(input_summary, use_container_width=True, hide_index=True)
