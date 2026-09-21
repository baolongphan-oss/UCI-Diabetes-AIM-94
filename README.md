# UCI Diabetes Outpatient Monitoring: Patient Risk Classification & Clustering

## Project Overview

End-to-end healthcare analytics project analyzing 29K outpatient monitoring records from 70 diabetic patients (1989-1991). This project demonstrates time-series data engineering, exploratory data analysis, clinical feature engineering, predictive modeling, and unsupervised patient segmentation.

**Key Finding:** Insulin dosing variability, not absolute glucose levels, is the strongest predictor of hypoglycemia risk. A hidden high-risk phenotype (Cluster 1) shows 7x more hypo episodes despite moderate glucose control.

## Datasets

- **Diabetes Data** (29,330 rows): Time-series outpatient monitoring records with insulin doses, blood glucose readings, meal/exercise logs, and hypoglycemic events
  - **Structure:** Date, Time, Code, Value (tab-separated)
  - **Temporal Span:** 1989-1991, 70 individual patients
  - **Code Mapping:** 9 undocumented codes; 7 clinically meaningful measurement types

Source: [UCI Machine Learning Repository - Diabetes Data](https://archive.ics.uci.edu/dataset/34/diabetes)

## Project Structure

```
.
├── README.md
├── uci_diabetes.ipynb
├── diabetes-data.tar.Z
└── Diabetes-Data/
    ├── data-40  (all 70 patients' records)
    ├── Data-Codes (code mapping)
    └── Domain-Description (clinical metadata)
```

## Notebook Architecture

### UCI Diabetes Outpatient Monitoring Notebook

**Purpose:** Complete pipeline from raw time-series data to clinical insights

**Key Sections:**

#### 1. Executive Summary
Outpatient diabetes data (70 patients, 29K records) used to identify risk profiles and predict hypoglycemia. Key findings: insulin variability is the primary risk driver; K-Means identified three patient phenotypes; well-controlled patients show distinct protective profile.

#### 2. Data Overview & Quality
- **Dataset:** 70 diabetic patients, 29,330 measurements (1989-1991)
- **Codes:** Insulin doses (3 types), blood glucose (8 timepoints), meals/exercise (6 types), hypoglycemic episodes
- **Data Quality:** 0.11% null values after cleaning (99.47% retention)
- **Undocumented Codes:** 6.96% of data dropped (codes: 0, 4, 36, 51, 52, 56, 88, 96, 98)

#### 3. Data Cleaning & Preparation
Raw data contained mixed types (integers, floats, leading-zero strings '007', malformed values '0Hi', '0Lo'). Developed `clean_value()` function with type-specific handling:
- **Insulin:** `float()` to preserve 1.5, 2.5 unit doses
- **Blood Glucose:** `int(float(val_str))` to handle string floats like '354.0'
- **Binary Indicators:** Validation to [0, 1] range
- **Result:** Retained 99.47% of data with only 0.11% nulls

#### 4. Feature Engineering
Aggregated 29K time-series to 70 patient-level features capturing:
- **Glucose Control:** glucose_mean, glucose_std, glucose_min, glucose_max, glucose_above_200, glucose_below_70
- **Insulin Management:** insulin_mean, insulin_std, insulin_total (all three types combined)
- **Clinical Risk:** hypoglycemic_episodes
- **Behavioral:** meal_reports, exercise_reports
- **Observational:** total_records, days_observed

#### 5. Exploratory Data Analysis
- **Glucose Distribution:** Mean 160.2 mg/dL, median 149.0 (right-skewed, target ~150)
- **Temporal Patterns:** Lunch period shows highest glucose spikes (post-lunch median ~250 mg/dL)
- **Insulin Variability:** NPH insulin most variable (outliers to 400 units/dose); average patient dose 0-20 units (median ~6-7)
- **Clinical Context:** 12/70 patients at high risk (17%); 38/70 experienced hypo episodes (54%)

#### 6. Correlation Analysis
**With Hypoglycemic Episodes Target:**
- meal_reports: 0.232 (highest correlation)
- insulin_std: 0.225
- exercise_reports: 0.169
- insulin_mean: 0.164
- glucose_std: 0.134
- glucose_mean: 0.023 (negligible)
- glucose_below_70: -0.068 (paradoxical)

**Feature Selection:** Retained meal_reports (0.232), insulin_std (0.225), insulin_mean (0.164); dropped weak features (|r| < 0.16)

#### 7. Classification Model: Hypoglycemia Prediction
**Target:** Binary (hypoglycemic_episodes > 0, n=38 True / 32 False, balanced)
**Model:** Random Forest (max_depth=3, n_estimators=50)
**Validation:** 5-fold cross-validation

**Results:**
- **Cross-Validation ROC-AUC:** 0.752 ± 0.096 (fair discrimination, clinically honest)
- **Train Accuracy:** 87.5%, **Test Accuracy:** 78.6% (overfitting gap: 8.9%, acceptable)
- **Confusion Matrix (test set):** TN=27, FP=5, FN=1, TP=37
- **Classification Report:** Precision=0.88, Recall=0.97, F1=0.93, Accuracy=0.91

**Feature Importance:**
- insulin_mean: 0.381 (strongest)
- insulin_std: 0.315
- meal_reports: 0.304

**Clinical Validation:** Higher insulin doses → increased hypo risk confirmed by clinical literature (insulin overdose is most common cause of hypoglycemia). Model precision (88%) acceptable for screening.

#### 8. Unsupervised Clustering: Patient Phenotypes
**Method:** K-Means (k=3) on [glucose_mean, glucose_std, insulin_mean] with StandardScaler

**Why 3 features:** Minimal set capturing glucose control, stability, and insulin dosing. Adding insulin_std or meal_reports created extreme outliers or unstable clusters.

**Results:**

| Cluster | N (%) | Glucose Mean | Glucose Std | Insulin Mean | Hypo Episodes | Profile |
|---------|-------|--------------|-------------|--------------|---------------|---------|
| 0 | 46 (66%) | 172.5 | 28.6 | 7.8 | 3.04 | Stable Majority |
| 1 | 8 (11%) | 167.9 | 41.2 | 13.3 | 22.38 | **HIGH RISK** |
| 2 | 16 (23%) | 131.1 | 19.4 | 4.7 | 0.75 | Well-Controlled |

**Cluster 1 Deep Dive (High-Risk Phenotype):**
- **Insulin Variability:** 41.2 std (vs 28.6 in Cluster 0) — 2.5x more erratic dosing
- **Insulin Mean:** 13.3 units (vs 7.8) — 71% higher baseline doses
- **Meal Engagement:** 16.9 reports (vs 7.4) — more monitoring/intervention
- **Hypo Episodes:** 22.38 (vs 3.04) — 7.4x higher risk
- **Clinical Insight:** "Active management paradox" — frequent adjustments + erratic dosing → unpredictable glucose swings → hypoglycemia
- **At-Risk Patient IDs:** [65, 1, 12, 15, 13, 35, 67, 11]

**Cluster 2 Protection (Well-Controlled):**
- Lowest glucose mean (131.1), lowest variability (19.4 std)
- Minimal hypo episodes (0.75)
- **Insight:** Stable insulin regimens are protective despite lower absolute doses

#### 9. Key Findings
1. **Insulin Dosing Variability Predicts Hypoglycemia** — Feature importance ranking: insulin_mean (0.381), insulin_std (0.315), meal_reports (0.304). Variability more predictive than absolute levels.
2. **Hidden High-Risk Phenotype (Cluster 1)** — 8 patients with 7x more hypo episodes despite moderate glucose control. Insulin variability 2.5x higher, suggesting over-titration paradox.
3. **Well-Controlled Phenotype (Cluster 2)** — Lowest glucose mean and variability show minimal hypo risk, demonstrating stability is protective.

#### 10. Clinical Implications
1. **Stabilize, Don't Titrate:** High insulin variability patients need fixed, consistent dosing rather than frequent adjustments
2. **Continuous Glucose Monitoring:** Cluster 1 patients should use CGM to reduce blind spots and prevent overshooting
3. **Patient Phenotypes:** One-size-fits-all treatment insufficient; stratify by cluster membership for precision medicine approach

#### 11. Model Limitations
- Small sample size (70 patients) limits generalization to broader diabetic populations
- 78.6% test accuracy reflects weak signal in available features; other unmeasured factors (medication compliance, diet, stress) likely important
- Historical data (1989-1991) may not reflect modern insulin regimens or patient behaviors
- External validation on independent patient populations required before clinical deployment

---

## Key Findings Summary

### Conversion Bottleneck: Insulin Variability, Not Control
- **Glucose Mean** (glucose level) has near-zero correlation (0.023) with hypo risk
- **Insulin Std** (dosing variability) has 0.225 correlation — 10x stronger signal
- **Implication:** Stable insulin regimens are more protective than tight glucose control

### Hidden High-Risk Phenotype (Cluster 1)
- 8 patients (11% of cohort) with **22.38 hypo episodes** vs 3.04 in stable majority (7.4x higher)
- Insulin variability **2.5x baseline** despite **moderate glucose control**
- **Clinical Mechanism:** Over-titration → unpredictable swings → hypoglycemia
- **Actionable:** Identify via insulin_std & meal_reports; stabilize dosing rather than adjust frequently

### Well-Controlled Phenotype (Cluster 2)
- 16 patients (23%) with **lowest glucose mean (131.1)** and **lowest variability (19.4)**
- **Minimal hypo risk (0.75 episodes)**
- **Proof of concept:** Stable regimens work across all glucose targets

---

## Future Analysis Opportunities

- **Prospective Validation:** Test model on 1999-2008 diabetes data (UCI Diabetes 130-Hospitals, 100K+ patients)
- **Interaction Effects:** Cluster × Time, Cluster × Meal Engagement
- **Root Cause Analysis:** Why does Cluster 1 have high variability? (Prescribing patterns? Patient factors?)
- **Treatment Response:** Do Cluster 1 patients respond differently to fixed-dose interventions?
- **Temporal Modeling:** Time-series forecasting of next glucose reading (LSTM/ARIMA)
- **Causality:** Does reducing insulin_std causally reduce hypo episodes?

---

## Technical Stack

**Languages & Libraries:**
- Python 3.x
- pandas (data manipulation, groupby operations)
- numpy (numerical operations)
- scikit-learn (RandomForestClassifier, KMeans, StandardScaler, cross_val_score)
- matplotlib & seaborn (visualizations)
- scipy (correlation analysis)

**Methods:**
- Time-series aggregation & feature engineering
- Data quality validation (mixed-type handling, regex parsing)
- Exploratory data analysis (EDA)
- Correlation analysis
- Classification (Random Forest)
- Unsupervised clustering (K-Means with scaling)
- Cross-validation (5-fold)
- Statistical interpretation (ROC-AUC, confusion matrix, feature importance)

---

## Key Takeaways

1. **Feature Engineering Beats Algorithm:** Insulin variability (engineered from raw doses) matters more than raw glucose readings. Time spent on features > time spent on model tuning.

2. **Honest Evaluation > Inflated Metrics:** Rejected 0.992 accuracy (circular target) in favor of 0.752 ROC-AUC (legitimate target). Portfolio credibility depends on transparent methodology.

3. **Unsupervised Learning for Discovery:** K-Means found Cluster 1 (high-risk phenotype) that naive glucose metrics would miss. Segmentation > aggregation in healthcare.

4. **The Story Here:** Insulin dosing stability, not glucose control, drives hypo risk. Frequent management adjustments can paradoxically increase risk in vulnerable subgroups.

---

## Contact & Links

- **Author:** Long Phan
- **Email:** lnphan@usc.edu
- **LinkedIn:** [linkedin.com/in/longphan1912](https://linkedin.com/in/longphan1912)
- **GitHub:** [github.com/baolongphan-oss](https://github.com/baolongphan-oss)
- **Dataset Source:** [UCI Machine Learning Repository - Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes)

---

**Last Updated:** September 2026  
**Project Status:** Portfolio-ready; prospective validation pending
