<p align="center">
  <img alt="University Paris Cité" src="https://img.shields.io/badge/University-Paris%20Cité-6f42c1?style=for-the-badge&logo=academia&logoColor=white">
  <img alt="Master ML for Data Science" src="https://img.shields.io/badge/Master-Machine%20Learning%20for%20Data%20Science-1976D2?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Practical Project" src="https://img.shields.io/badge/Project-Human%20Activity%20Recognition-FF9800?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Academic Year" src="https://img.shields.io/badge/Year-2025%2F2026-009688?style=for-the-badge&logo=googlecalendar&logoColor=white">
</p>

---

<p align="center">
  <strong> Master 2 Machine Learning for Data Science</strong><br>
  <strong>Université Paris Cité</strong> — UFR Sciences Fondamentales et Biomédicales
</p>

---

###  Project Information

| **Category**       | **Details** |
|--------------------|-------------|
| **University**     | Université Paris Cité |
| **Master Program** | Machine Learning for Data Science |
| **Project Type**   | Human Activity Recognition — Unsupervised Learning |
| **Supervisor**     | Dr. Allou Samé *(Université Gustave Eiffel — Classification Automatique)* |
| **Student**        | Hamady GACKOU *(Government of France Excellence Scholar, MEAE)* |
| **Academic Year**  | 2025–2026 |

---

#  Human Activity Recognition — Unsupervised Classification Project

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ML-lightgrey)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

---

##  Overview

**Objective:**  
This project explores **unsupervised learning techniques** to recognize human physical activities using smartphone sensor data.  
The target is to build **interpretable, robust, and scalable representations** of motion patterns without labeled supervision.

**Detected Activities:**
- Walking  
- Going upstairs  
- Going downstairs  
- Sitting  
- Standing  
- Lying down  

---

##  Dual Analytical Strategy — “Two Complementary Approaches”

| **Approach** | **Principle** | **Data Representation** | **Core Algorithms** |
|--------------|---------------|--------------------------|---------------------|
| **1. Direct (Distance-Based)** | Uses full temporal sequences (128×9) to compare motion via temporal distances (e.g. DTW, Correlation). | Tensor (347×128×9) | DTW-KMeans (DBA), PAM, Hierarchical (CAH) |
| **2. Feature-Based (Transformation)** | Extracts compact statistical & spectral descriptors (mean, std, skewness, kurtosis, FFT, AR coefficients). | Matrix (347×p) | KMeans, GMM, SOM, Ward Hierarchical |

>  **Strategic Insight:** Combining temporal fidelity (Direct) with interpretability (Feature-Based) yields a **hybrid analytical framework** for motion understanding — valuable in embedded AI, wearable analytics, and health monitoring.

---

##  Exploratory Data Analysis (EDA)

**Goals:**
- Validate data integrity  
- Explore structure, variance, and frequency composition  
- Prepare clean and meaningful inputs for clustering  

**Dataset Summary:**
- 347 temporal windows (~2.56 s each)  
- 9 sensors per window (`accm_x, accm_y, accm_z, acce_x, acce_y, acce_z, vit_x, vit_y, vit_z`)  
- Shapes:  
  - `X`: (347, 128, 9) — raw tensor  
  - `Z`: (347, 54) — feature matrix  
  - `y`: (347,) — ground truth labels (for evaluation only)

**Key EDA Results:**
- No missing or infinite values.  
- Statistically stable variables (consistent mean and variance).  
- PCA, UMAP, and t-SNE projections confirm **nonlinear separability** between activities.  
- FFT analysis isolates **rhythmic frequencies (1–3 Hz)** for dynamic activities.  
- Strong intra-sensor correlations justify dimensionality reduction.

>  **Strategic Value:** Demonstrates mastery of data preprocessing, signal analysis, and statistical representation — transferable to any sensor-based or time-series project.

---

##  Clustering Methodology

**Algorithms Evaluated:**
- Partition-based: *KMeans, GMM, Ward Hierarchical, Birch*  
- Density-based: *DBSCAN, HDBSCAN, OPTICS*  
- Neural topology: *Self-Organizing Map (SOM)*  
- Temporal methods: *DTW-KMeans, KShape*  

**Dimensionality Reductions Used:**
- PCA (linear structure)  
- UMAP (manifold preservation)  
- t-SNE (local neighborhood exploration)

---

## Quantitative Evaluation

| **Metric** | **Best Configuration** | **Insight** |
|-------------|------------------------|--------------|
| **Silhouette Score** | HDBSCAN + PCA → *0.462* | Compact, coherent clusters |
| **Adjusted Rand Index (ARI)** | KMeans + t-SNE → *0.98* | Perfect label alignment |
| **Normalized Mutual Information (NMI)** | DTW-KMeans → *0.95* | Strong consistency with true activities |
| **Calinski–Harabasz** | GMM / Ward → *>90,000* | Stable geometric cohesion |

> **Strategic Insight:**  
> - The **Feature-Based framework** ensures interpretability and stability (ideal for real-time monitoring).  
> - The **Time-Series framework** excels at temporal continuity and phase preservation (critical for biomechanical or sports applications).

---

##  Visual Analytics

- **Cluster visualization** using PCA/UMAP projections.  
- **Hierarchical dendrograms** showing activity-level separability.  
- **Confusion matrices** (clusters vs. true labels).  
- **Centroid signal plots** highlighting mean motion profiles.  
- **Correlation heatmaps** linking feature and temporal representations.

> **Strategic Impact:** Visual interpretability bridges machine decisions and human reasoning — a key requirement in explainable AI (XAI).

---

## Key Results Summary

| **Approach** | **Strengths** | **Applications** |
|---------------|---------------|------------------|
| **Feature-Based** | Compact, interpretable, robust to noise | Edge computing, health monitoring |
| **Time-Series (DTW)** | Preserves temporal rhythm & dynamics | Gait analysis, activity tracking |
| **Hybrid Potential** | Combines both paradigms | Smart sensors, multimodal fusion |

---

##  Technical Environment

- **Languages:** Python 3.10  
- **Core Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn, Tslearn, UMAP-learn, SciPy  
- **Tools:** Jupyter Notebook, Git, PowerShell setup automation  
- **Environment Setup:**

```powershell
# Step 1: Setup virtual environment and kernel
.\setup\setup.ps1

# Step 2: Activate environment
.\har_env\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt


