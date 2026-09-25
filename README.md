# UCI Diabetes Outpatient Monitoring: Hypoglycemia Risk Classification & Clustering

## Project Overview

End-to-end healthcare analytics project on 29K outpatient monitoring records from 70 diabetic patients (1989-1991). The project covers messy clinical time-series cleaning, patient-level feature engineering, hypoglycemia risk classification, and K-Means patient segmentation.

**Key Finding:** Insulin dosing, not average glucose, tracks hypoglycemia risk. Mean glucose is essentially uncorrelated with hypoglycemic episodes (r = 0.02), while insulin dose variability (r = 0.23) and mean insulin dose (the model's top feature) carry the signal. Clustering surfaced a high-risk group whose glucose profile matches the majority, but whose insulin dosing is 2.5x more variable.

## Dataset

- **Diabetes Data** (29,330 records): Outpatient monitoring logs of insulin doses, blood glucose readings, meal and exercise events, and hypoglycemic symptoms
  - **Format:** Date, Time, Code, Value (tab-separated), one file per patient
  - **Span:** 1989-1991, 70 patients

Source: [UCI Machine Learning Repository: Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes)

## Project Structure

```
.
├── README.md
├── uci_diabetes.ipynb
├── diabetes-data.tar.Z
└── Diabetes-Data/
    ├── data-01 ... data-70   (one file per patient)
    ├── Data-Codes            (code mapping)
    └── Domain-Description    (clinical metadata)
```

## Approach

### 1. Data Cleaning
Raw values mixed integers, floats, leading-zero strings (`'007'`), and malformed entries (`'0Hi'`, `'0Lo'`). A `clean_value()` function handles each measurement type:
- **Insulin:** `float()` to keep half-unit doses (1.5, 2.5)
- **Blood glucose:** `int(float(val_str))` for string floats like `'354.0'`
- **Binary indicators:** validated to 0 or 1

Undocumented codes (0, 4, 36, 51, 52, 56, 88, 96, 98) were dropped (6.96% of records). Value cleaning left only 0.11% nulls (99.47% retention).

### 2. Feature Engineering
Aggregated 29K time-series records into one row per patient (70 rows):
- **Glucose:** mean, std, min, max, readings above 200 and below 70 mg/dL
- **Insulin:** mean, std, total (all three insulin types combined)
- **Clinical risk:** hypoglycemic episode count
- **Behavioral:** meal and exercise reports
- **Observation window:** total records, days observed

### 3. Target Selection: Catching a Circular Target
The first target, "poor glucose control" (mean glucose > 200 mg/dL, 12 of 70 patients), was built from the same glucose features available to predict it. Correlation analysis confirmed the circularity (glucose_mean r = 0.72 with the target), while every independent feature was weak (|r| < 0.22). The target was dropped in favor of **hypoglycemia risk** (any hypoglycemic episode: 38 yes, 32 no), which is clinically meaningful and not defined by the model's inputs.

### 4. Classification: Hypoglycemia Risk
- **Features:** insulin_mean, insulin_std, meal_reports (strongest independent correlations with the target)
- **Model:** Random Forest (`max_depth=3`, `n_estimators=50`), kept shallow to limit overfitting on 70 patients
- **Validation:** 5-fold cross-validation plus an 80/20 train/test split

| Metric | Result |
|---|---|
| **Cross-validated ROC-AUC** | **0.752 ± 0.096** |
| Train accuracy | 87.5% |
| Test accuracy (14 patients) | 78.6% |

**Feature importance:** insulin_mean (0.381), insulin_std (0.315), meal_reports (0.304)

A cross-validated ROC-AUC of 0.75 means fair discrimination: useful as a screening signal, not a diagnostic tool. The wide ± 0.096 reflects the small sample.

*Note: The notebook's confusion matrix is computed on all 70 patients after fitting on those same patients, so it shows in-sample fit rather than performance on unseen patients. Cross-validated ROC-AUC is the reliable measure here.*

### 5. Clustering: Patient Phenotypes
**Method:** K-Means (k = 3) on standardized glucose_mean, glucose_std, insulin_mean, hypoglycemic_episodes, and days_observed.

| Cluster | Patients | Glucose Mean | Glucose Std | Insulin Mean | Insulin Std | Hypo Episodes | Meal Reports |
|---|---|---|---|---|---|---|---|
| 0: Stable Majority | 46 (66%) | 172.5 | 86.3 | 7.6 | 5.0 | 3.04 | 7.4 |
| 1: High Risk | 8 (11%) | 167.9 | 87.9 | 13.3 | 12.5 | 22.38 | 16.9 |
| 2: Well-Controlled | 16 (23%) | 131.1 | 52.4 | 7.9 | 5.8 | 0.75 | 2.4 |

**Cluster 1 (High Risk) vs. Cluster 0 (Majority):**
- **Glucose looks the same:** mean 167.9 vs. 172.5, variability 87.9 vs. 86.3
- **Insulin dosing is different:** 75% higher mean dose (13.3 vs. 7.6) and **2.5x higher dose variability** (12.5 vs. 5.0)
- **More logging activity:** 2.3x more meal reports (16.9 vs. 7.4)
- **Hypoglycemic episodes:** 7.4x higher (22.38 vs. 3.04). Episode count was a clustering input, so this separation is partly by design.

The notable result is that **insulin variability and meal reports were not clustering inputs**, yet both clearly distinguish the high-risk group. A glucose-only view would not flag these patients.

**Cluster 2 (Well-Controlled):** Lowest glucose mean (131.1) and variability (52.4), and the fewest hypoglycemic episodes (0.75). It also has the fewest meal reports (2.4), so part of its apparent stability may reflect less logging rather than better control.

## Key Findings

1. **Insulin dosing tracks hypoglycemia risk; average glucose doesn't.** Mean glucose has near-zero correlation with episodes (r = 0.02). Mean insulin dose is the model's strongest feature, consistent with clinical research identifying excess insulin as the most common cause of hypoglycemia.
2. **A high-risk group hides behind normal-looking glucose.** Cluster 1's glucose profile matches the majority, but its insulin doses are higher and 2.5x more variable.
3. **Honest targets beat impressive-looking ones.** The first target was circular and was dropped before modeling. The final model's 0.75 ROC-AUC is modest but legitimate.

## Clinical Implications (Hypotheses to Test)

These come from an observational sample of 70 patients, so they're hypotheses, not recommendations:
- Patients with highly variable insulin dosing may warrant closer hypoglycemia monitoring, even when average glucose looks normal.
- Dosing consistency may matter alongside glucose targets. Whether stabilizing doses reduces episodes would require a controlled study.
- Grouping patients by dosing behavior may identify risk that glucose averages miss.

## Limitations

- **Small sample:** 70 patients, with only 14 in the test split, so metrics carry wide uncertainty.
- **In-sample confusion matrix:** See the note in the Classification section.
- **Logging behavior as a confounder:** Meal reports and hypoglycemic episodes both depend on how actively a patient logged. Patients who record more may simply show more events.
- **Clustering includes the outcome:** Hypoglycemic episodes were a clustering input, which partly drives the high-risk group's separation.
- **Historical data:** 1989-1991 insulin regimens and monitoring practices differ from today's.
- **Correlation, not causation:** No finding here shows that changing dosing patterns would change outcomes.

## Future Analysis

- **Cross-validated precision and recall:** Use `cross_val_predict` so every patient is scored by a model that never saw them.
- **Re-run clustering without hypoglycemic episodes:** Test whether the high-risk group still emerges from glucose and insulin alone.
- **Control for logging volume:** Normalize counts by days observed or total records.
- **Validate on a larger cohort:** The 70-patient sample limits generalization.
- **Time-series modeling:** Forecast near-term hypoglycemia risk from recent insulin and glucose readings.

## Technical Stack

**Tools:** Python, pandas, NumPy, scikit-learn (RandomForestClassifier, KMeans, StandardScaler, PCA, cross_val_score), matplotlib, seaborn, Jupyter

**Methods:** Clinical time-series cleaning, patient-level feature engineering, correlation analysis, Random Forest classification, 5-fold cross-validation, K-Means clustering, PCA visualization

## Key Takeaways

1. **Feature engineering mattered more than the algorithm.** Insulin variability, engineered from raw dose logs, carried more signal than raw glucose readings.
2. **Check the target before trusting the model.** A target built from the model's own inputs will always look predictable.
3. **Clustering can reveal risk that averages hide.** The high-risk group only stands out once insulin behavior is considered.

## Contact

- **Author:** Long Phan
- **Email:** lnphan@usc.edu
- **LinkedIn:** [linkedin.com/in/longphan1912](https://www.linkedin.com/in/longphan1912)
- **GitHub:** [github.com/baolongphan-oss](https://github.com/baolongphan-oss)

---

**Last Updated:** September 2026
