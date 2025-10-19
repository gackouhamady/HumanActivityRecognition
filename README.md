<p align="center">
  <img alt="University Paris Cité" src="https://img.shields.io/badge/University-Paris%20Cité-6f42c1?style=for-the-badge&logo=academia&logoColor=white">
  <img alt="Master ML for Data Science" src="https://img.shields.io/badge/Master-Machine%20Learning%20for%20Data%20Science-1976D2?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Practical Project" src="https://img.shields.io/badge/Project-Practical%20Lab-FF9800?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Academic Year" src="https://img.shields.io/badge/Year-2025%2F2026-009688?style=for-the-badge&logo=googlecalendar&logoColor=white">
</p>

---

<p align="center">
  <strong>🎓 Master 2 Machine Learning for Data Science</strong>
</p>

---

<p align="center">

### 📊 Project Information  

| **Category**       | **Details**                           |
|--------------------|---------------------------------------|
| **University**     | University Paris Cité                 |
| **Master Program** | Machine Learning for Data Science     |
| **Project Type**   | Human Activity Recognition (Practical Project) |
| **Supervisor**     | Allou Samé                             |
| **Student**        | Hamady GACKOU                          |
| **Academic Year**  | 2025/2026                              |

</p>

---

# Human Activity Recognition - Unsupervised Classification Project

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ML-lightgrey)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

---

## Project Overview

**Objective:**  
This project aims to apply **unsupervised classification algorithms** to detect human activities from smartphone sensor data. The activities include:
- Walking  
- Climbing stairs  
- Descending stairs  
- Sitting  
- Standing  
- Lying down  

**Data Description:**  
- Collected from a smartphone during a controlled experiment.  
- **9 sensor variables** measured every 0.02 seconds:  
  - Accelerations: `accm_x`, `accm_y`, `accm_z`  
  - Estimated accelerations (without gravity): `acce_x`, `acce_y`, `acce_z`  
  - Velocities: `vit_x`, `vit_y`, `vit_z`  
- Data segmented into **temporal windows** of 128 observations (~2.5 seconds).  
- Total windows: 347  
- True activity labels are available **only for evaluation purposes**.

---

## Methods

- **Algorithms applied**:
  - K-means
  - Hierarchical Clustering (CAH)
  - DBSCAN / HDBSCAN
  - Optional: Spectral Clustering for non-linear separations

- **Feature extraction per window** (optional but recommended):
  - Mean, variance, minimum, maximum
  - Higher-order moments
  - Autoregressive coefficients
  - Fourier coefficients

- **Evaluation metrics**:
  - Silhouette score
  - Adjusted Rand Index (ARI)
  - Cluster visualizations

---

## Project Pipeline

1. **Data Loading & Exploration**  
2. **Preprocessing & Feature Extraction**  
3. **Dimensionality Reduction (optional: PCA, t-SNE)**  
4. **Clustering with multiple algorithms**  
5. **Evaluation & Comparison of clusters**  
6. **Results Interpretation and Visualization**  
7. **Report Writing and Figures Preparation**

## Project Setup - Human Activity Recognition

```powershell
# Step 1: Setup virtual environment and Jupyter kernel
.\setup\setup.ps1
```
```powershell
# Step 2: Activate the virtual environment
.\har_env\Scripts\activate
```

```powershell
# Step 3: Install project dependencies
pip install -r requirements.txt
```


