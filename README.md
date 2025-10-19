<p align="center">
  <img alt="University Paris Cité" src="https://img.shields.io/badge/University-Paris%20Cité-6f42c1?style=for-the-badge&logo=academia&logoColor=white">
  <img alt="Module Data Engineering" src="https://img.shields.io/badge/Course-Data%20Engineering-1976D2?style=for-the-badge&logo=databricks&logoColor=white">
  <img alt="Practical Lab" src="https://img.shields.io/badge/Type-Practical%20Lab-FF9800?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Day 1" src="https://img.shields.io/badge/Session-Day%201%20of%20Course-009688?style=for-the-badge&logo=googlecalendar&logoColor=white">
</p>

---

<p align="center">
  <strong>🎓 Master 2 Machine Learning for Data Science</strong>
</p>

---

<p align="center">

### 📊 Course Information  

| **Category**      | **Details**                          |
|-------------------|--------------------------------------|
| **University**    | University Paris Cité                |
| **Teaching Unit** | Data Engineering (Practical Labs)    |
| **Session**       | Day 1 of the course                  |
| **Instructor**    | Amine Ferddjaoui                     |
| **Student**       | Hamady GACKOU                        |
| **Supervisor**    | Allou Samé                           |
| **Academic Year** | 2025/2026                             |

</p>

---

# Human Activity Recognition - Unsupervised Classification Project

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ML-lightgrey)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

---

## Project Overview

**Title:** Detection of Human Activities using Smartphones  
**Master Program:** Master 2 Machine Learning for Data Science, Université Paris Cité  
**Student:** Hamady GACKOU  
**Supervisor:** Allou Samé  
**Academic Year:** 2025/2026  

**Objective:**  
The objective of this project is to apply **unsupervised classification algorithms** to identify human activities from smartphone sensor data. The activities to detect include:
1. Walking  
2. Climbing stairs  
3. Descending stairs  
4. Sitting  
5. Standing  
6. Lying down  

---

## Data Description

- Collected via smartphone sensors during a human activity experiment.  
- **9 variables** measured every 0.02 seconds:
  - Accelerations: `accm_x`, `accm_y`, `accm_z`
  - Estimated accelerations (without gravity): `acce_x`, `acce_y`, `acce_z`
  - Velocities: `vit_x`, `vit_y`, `vit_z`  
- Data is divided into **temporal windows of 128 observations** (~2.5 seconds per window).  
- Total windows: 347  
- True labels (`lab`) are available for evaluation purposes only.  

---

## Methods

- **Unsupervised algorithms** applied:
  - K-means  
  - Hierarchical Clustering (CAH)  
  - DBSCAN / HDBSCAN  
  - Optional: Spectral Clustering
- **Features extracted per window** (optional but recommended):
  - Mean, variance, minimum, maximum
  - Higher-order moments (skewness, kurtosis)
  - Autoregressive coefficients
  - Fourier coefficients  

---

## Deliverables

- A **Python notebook** containing the complete analysis.  
- A **PDF report** (≤15 pages) summarizing methodology, results, and discussion.  

---

## Project Pipeline

1. **Data Loading & Exploration**  
   Load the raw sensor data, check for missing values, and visualize initial trends.

2. **Preprocessing & Feature Extraction**  
   Standardize variables and compute features per temporal window.

3. **Dimensionality Reduction (optional)**  
   Apply PCA or t-SNE for visualization and noise reduction.

4. **Unsupervised Classification**  
   Apply K-means, CAH, DBSCAN/HDBSCAN, and optionally Spectral Clustering.

5. **Evaluation & Comparison of Methods**  
   Metrics: silhouette score, adjusted Rand index (ARI), cluster heatmaps.

6. **Results Interpretation**  
   Compare clusters with true labels (if available) and analyze patterns.

7. **Report Writing**  
   Summarize methods, results, and discussion in a clear, reproducible format.

---
