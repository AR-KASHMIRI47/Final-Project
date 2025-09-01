
**Thyroid-Function Classification from Routine Lab Measurements** 

---

# Overview
This notebook implements an end-to-end machine learning pipeline for thyroid disease classification. It explores class imbalance handling using SMOTE, performs feature selection with Random Forest importance, and compares multiple models (Logistic Regression, SVM, Random Forest, Gradient Boosting) across baseline, feature-selected, and balanced datasets.

---

# Dataset
- **Path/URL:** yasserhessein/thyroid-disease-data-set (Kaggle)
- **Target column:** `binaryclass`
- **Feature column(s):** age, sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery, i131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, tsh_measured, tsh, t3_measured, t3, tt4_measured, tt4, t4u_measured, t4u, fti_measured, fti, referral_source  
- **Feature count/types:** 30 columns (mixed numerical + categorical)

---

# Features & Preprocessing
- Standardized column names to lowercase with underscores.  
- Dropped low-variance columns (`tbg`, `tbg_measured`).  
- Replaced “?” with mode values for categorical fields.  
- Imputed missing numerical values with the median.  
- Encoded categorical/boolean variables into numeric form.  
- Applied SMOTE for class imbalance correction.  
- Selected top 20 features based on Random Forest importance.

---

# Models
- **Logistic Regression** (`solver='liblinear'`, `max_iter=1000`, `random_state=42`)  
- **Support Vector Machine (SVC)** (`probability=True`, `random_state=42`)  
- **Random Forest Classifier** (`random_state=42`)  
- **Gradient Boosting Classifier** (`random_state=42`)

---

# Evaluation
- **Metrics:** accuracy, precision, recall, F1-score (macro), ROC-AUC.  
- **Visualizations:** confusion matrices, ROC curves, feature importance plots, SMOTE distribution plots.  
- **Tuning:** feature selection with Random Forest importance; model comparison across baseline, selected features, and SMOTE-balanced data.

---

# Environment & Requirements
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn.  
- **Install example:**
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
  ```
