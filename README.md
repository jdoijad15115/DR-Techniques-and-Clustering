# DR-Techniques-and-Clustering and 🛰️ EuroSAT Image Analysis 

## 📌 Project Title:
**Performance Analysis of Dimensionality Reduction & Clustering Techniques on EuroSAT Satellite Imagery**

---

## 📖 Overview

This project applies **Dimensionality Reduction (DR)** and **Clustering** methods on the **EuroSAT** dataset to analyze land use and land cover patterns from Sentinel-2 satellite images. The workflow includes:
- Image loading and preprocessing
- Feature extraction via deep learning (ResNet50)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Clustering (KMeans, Hierarchical, DBSCAN)
- Visualization and evaluation of clusters

---

## 📂 Files in This Repository

- `Clustring.ipynb` – Clustering methods & evaluations
- `DR_Techniques.ipynb` – Feature extraction and DR visualization
- `README.md` – Project summary

---

## 📊 Dataset Details

- **Name:** [EuroSAT RGB](https://zenodo.org/record/7711810)
- **Images:** 27,000 satellite images (RGB)
- **Size:** 64x64 px
- **Classes (10):**
  - 🏘️ Residential  
  - 🏭 Industrial  
  - 🛣️ Highway  
  - 🌊 River  
  - 🌳 Forest  
  - 🌾 Pasture  
  - 🌱 Herbaceous vegetation  
  - 🌴 Permanent crop  
  - 🏖️ Sea/Lake  
  - 🚜 Agricultural land

---

## ⚙️ Steps Performed

### 🧾 Data Preparation
- Downloaded and unzipped EuroSAT RGB dataset
- Undersampled to 2000 images/class for balanced computation
- Visualized folder structure and data samples

### 🧠 Feature Extraction
- Used **ResNet50** pretrained model to extract image embeddings
- Flattened and stored feature vectors for further analysis

### 📉 Dimensionality Reduction
- Applied:
  - **PCA** (Principal Component Analysis)
  - **t-SNE**
  - **UMAP**
- Visualized data in 2D for cluster separation

### 🔗 Clustering Techniques
- **KMeans**
- **Agglomerative (Hierarchical) Clustering**
- **DBSCAN (Density-Based Clustering)**
- Plotted clusters and analyzed groupings with color-coded labels

### 📈 Evaluation
- Silhouette Score
- Cluster plots (with DR techniques)
- Dendrogram (for hierarchical clustering)

---

## 🛠️ Tech Stack

- Python (Jupyter/Colab)
- Libraries:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `sklearn`, `scipy`, `umap-learn`
  - `tensorflow.keras.applications` (for ResNet50)
  - `PIL`, `os`, `shutil`, `glob`

---

## 📌 Key Insights

- Feature extraction via deep learning significantly improved cluster quality.
- Dimensionality reduction enabled effective 2D visualization.
- LDA provided better separation compared to t-SNE, MDS, PCA, SVD in this context.
- KMeans performed well with silhouette ~0.52 post-DR.
- DBSCAN was more sensitive to parameter tuning, while Hierarchical gave interpretable tree structures.

---

## 📬 How to Run

1. Upload both notebooks (`DR_UMLLAB.ipynb`, `Clustring_UMLLAB.ipynb`) to Google Colab.
2. Run DR notebook to:
   - Load dataset
   - Extract features with ResNet50
   - Reduce dimensionality
3. Run Clustering notebook to:
   - Apply clustering methods
   - Visualize and evaluate clusters

---

## 🧠 Learnings

- Integration of deep learning with unsupervised ML
- Comparison of DR techniques for real-world satellite data
- Importance of evaluation metrics and visuals in cluster analysis

