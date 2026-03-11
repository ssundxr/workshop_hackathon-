# ❤️ Heart Disease Prediction App
### Pragyan AI Hackathon — Classification Project

---

## 🚀 Quick Start (Run Locally)

### 1. Clone / Download the project
```bash
# If using git
git clone <your-repo-url>
cd heart_disease_app
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** 🎉

---

## ☁️ Deploy to Streamlit Cloud (Free, 5 minutes)

1. Push your project to a **GitHub repo**
2. Go to **https://share.streamlit.io**
3. Click **"New app"**
4. Select your repo, branch (`main`), and file (`app.py`)
5. Click **Deploy** ✅

Your app will be live at:
`https://<your-username>-<repo>-app-<hash>.streamlit.app`

---

## 🗂 Project Structure

```
heart_disease_app/
├── app.py              ← Main Streamlit app (all-in-one)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 📋 App Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Dataset overview, feature descriptions |
| 📊 EDA | Charts: distributions, correlations, scatter plots |
| 🤖 Model Training | Confusion matrix, ROC curve, feature importance |
| 🔮 Predict | Interactive patient risk predictor |

---

## 🧠 ML Pipeline

```
Raw Data (303 patients, 13 features)
    ↓
Train/Test Split (80/20, stratified)
    ↓
StandardScaler normalization
    ↓
Logistic Regression  +  Random Forest (200 trees)
    ↓
Evaluation: Accuracy, ROC-AUC, Precision, Recall, F1
    ↓
Streamlit interactive prediction UI
```

---

## 📊 Dataset

**Source:** [UCI Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Features (13):**
- `age`, `sex`, `cp` (chest pain), `trestbps` (blood pressure)
- `chol` (cholesterol), `fbs` (fasting blood sugar), `restecg`
- `thalach` (max heart rate), `exang`, `oldpeak`, `slope`, `ca`, `thal`

**Target:** `target` — 0 = No Disease, 1 = Disease

---

## 📦 Tech Stack

- **Python 3.9+**
- **Streamlit** — Web app framework
- **scikit-learn** — ML models
- **pandas / numpy** — Data processing
- **matplotlib / seaborn** — Visualizations

---

## 💡 Hackathon Tips

- The app auto-loads data from a public URL (no manual download needed)
- Falls back to synthetic data if network is unavailable
- Both models are trained fresh on every session load (cached for speed)
- Add SHAP explainability for bonus points!

---

*Built for Pragyan AI Hackathon | Grow with Gyan 🚀*
