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
This project aims to apply **unsupervised classification algorithms** to identify human activities from smartphone sensor data. The activities to detect include:
1. Walking  
2. Climbing stairs  
3. Descending stairs  
4. Sitting  
5. Standing  
6. Lying down  

**Data Description:**  
- Collected via smartphone sensors during a human activity experiment.  
- 9 variables measured every 0.02 seconds:
  - Accelerations: `accm_x`, `accm_y`, `accm_z`
  - Estimated accelerations (without gravity): `acce_x`, `acce_y`, `acce_z`
  - Velocities: `vit_x`, `vit_y`, `vit_z`  
- Data divided into **temporal windows** of 128 observations (~2.5 seconds).  
- Total windows: 347  
- True labels (`lab`) available for evaluation purposes only.

**Methods:**  
- Unsupervised algorithms applied:
  - K-means
  - Hierarchical Clustering (CAH)
  - DBSCAN / HDBSCAN
  - Optional: Spectral Clustering
- Features per window may include:
  - Mean, variance, min, max
  - Higher-order moments
  - Autoregressive coefficients
  - Fourier coefficients

**Deliverables:**  
- A **Python notebook** containing the complete analysis.  
- A **PDF report** (≤15 pages) summarizing methodology, results, and discussion.  

---

## Project Pipeline

1. **Data Loading & Exploration**
2. **Preprocessing & Feature Extraction**
3. **Dimensionality Reduction (optional)**
4. **Unsupervised Classification**
5. **Evaluation & Comparison of Methods**
6. **Results Interpretation**
7. **Report Writing**

---

> This Markdown serves as the **project documentation template**, and can be included at the top of the notebook or README file for a clear, professional overview.
