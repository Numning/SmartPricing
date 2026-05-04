# 📱 Smart Pricing — Predicting Smartphone Prices from Specifications

> **ITCS 227 — Introduction to Data Science | Semester 2/2025**

An end-to-end machine learning pipeline that predicts smartphone prices from hardware specifications, detects overpriced/underpriced phones, and deploys an interactive web application for consumers and manufacturers.

---

## 📁 Project Files

```
Project/
├── smartphone_price_prediction.ipynb   ← Main analysis notebook (full pipeline)
├── streamlit_app.py                    ← Interactive web application
├── best_model.pkl                      ← Pre-trained Extra Trees model (pickle)
├── plots/                              ← 20 generated visualizations
│   ├── 01_price_distribution.png
│   ├── 02_brand_comparison.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_yearly_trends.png
│   ├── 05_feature_vs_price.png
│   ├── 06_brand_premium.png
│   ├── 07_ram_by_brand.png
│   ├── 08_regression_results.png
│   ├── 09_feature_importance.png
│   ├── 10_deal_detector.png
│   ├── 11_classification_results.png
│   ├── 12_price_tiers.png
│   ├── 13_elbow_method.png
│   ├── 14_clustering_results.png
│   ├── 15_cross_country_prices.png
│   ├── 16_country_markup_heatmap.png
│   ├── 17_learning_curves.png
│   ├── 18_partial_dependency.png
│   ├── 19_clustering_comparison.png
│   └── 20_silhouette_comparison.png
└── README.md                           ← This file
```

---

## 🎯 Project Objectives

1. **Price Prediction** — Predict the fair market USD price of any smartphone from its hardware specs using regression models.
2. **Price Tier Classification** — Classify phones into Budget / Mid-Range / Premium / Ultra-Premium tiers.
3. **Deal Detector** — Flag phones priced >25% above or below their predicted fair value.
4. **Feature Importance** — Identify which specifications drive smartphone pricing the most.
5. **Cross-Country Analysis** — Compare prices across 5 countries (USA, India, China, Pakistan, Dubai).
6. **Clustering** — Discover natural market segments using KMeans, DBSCAN, and GMM.

---

## 📊 Dataset

| Property       | Value                                                        |
|----------------|--------------------------------------------------------------|
| Source         | [Kaggle — Mobiles Dataset 2025](https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025) |
| Author         | Abdul Malik (2025)                                           |
| License        | Apache 2.0                                                   |
| Raw size       | 930 rows × 15 columns                                        |
| After cleaning | 829 smartphones (tablets removed)                            |
| Brands         | 19 (Apple, Samsung, Xiaomi, Oppo, Vivo, Honor, Google, etc.)|
| Years          | 2014 – 2025                                                  |
| Price markets  | USA, India, China, Pakistan, Dubai                           |

The dataset is downloaded automatically via `kagglehub` when the notebook or Streamlit app is run for the first time.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip

### Install Dependencies

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn streamlit kagglehub
```

### Kaggle API (for dataset download)
The dataset downloads automatically using `kagglehub`. If you have not used Kaggle before, set up your API credentials:

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token
2. Place the downloaded `kaggle.json` in `~/.kaggle/kaggle.json`

---

## 🚀 Running the Project

### 1. Run the Main Analysis Notebook

Open and run all cells in Jupyter:

```bash
jupyter notebook smartphone_price_prediction.ipynb
```

This will:
- Download the dataset automatically
- Preprocess and engineer features
- Run EDA and generate all 20 plots (saved to `plots/`)
- Train and evaluate 8 regression models, 3 classifiers, 3 clustering algorithms
- Export the best model as `best_model.pkl`

### 2. Run the Streamlit Web Application

```bash
python3 -m streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** The app loads and trains models on first run (may take ~30 seconds). Subsequent loads use the cached model.

---

## 🌐 Streamlit App — Two Modes

### 🛒 Customer Mode
> "Is My Phone a Good Deal?"

Enter a phone's **specifications** and its **listed price (USD)**. The app will:
- Predict the fair market price based on hardware specs
- Show the price difference as a percentage
- Flag the phone as: **OVERPRICED** / **SLIGHTLY OVERPRICED** / **FAIRLY PRICED** / **GOOD DEAL** / **EXCELLENT DEAL**
- Display similar phones from the dataset for comparison

### 💼 Company Mode
> "What Should I Price This Phone?"

Enter a phone's **specifications only** (no brand bias). The app will:
- Predict the recommended selling price based on pure hardware value
- Suggest three pricing strategies: Aggressive (−15%), Market Rate, Premium (+15%)
- Show feature importance for hardware-based pricing
- Compare the phone against others in the same price tier

---

## 🤖 Models & Techniques

### Regression (Price Prediction)

| Model                   | MAE   | RMSE  | R²   | CV R² |
|-------------------------|-------|-------|------|-------|
| Linear Regression        | $116  | $173  | 0.68 | 0.68  |
| Ridge Regression         | $116  | $173  | 0.68 | 0.68  |
| Poly-2 Ridge             | $85   | $127  | 0.83 | 0.80  |
| Random Forest            | $78   | $118  | 0.85 | 0.85  |
| Gradient Boosting        | $69   | $106  | 0.88 | 0.87  |
| HistGradientBoosting     | $68   | $106  | 0.88 | 0.87  |
| **Extra Trees ⭐ (Best)**| $74   | $113  | **0.91** | **0.89** |
| Tuned Random Forest      | $75   | $116  | 0.86 | 0.85  |

⭐ = Best model

### Classification (Price Tier)

| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| Logistic Regression | ~0.66    | ~0.65    |
| Decision Tree        | ~0.76    | ~0.76    |
| **Random Forest**    | **~0.80**| **~0.79**|

### Clustering

| Method       | Silhouette Score | Clusters |
|--------------|-----------------|----------|
| **KMeans**   | **0.2830**      | 4        |
| DBSCAN       | 0.1512          | 5        |
| GMM          | 0.2397          | 2        |

---

## 🔑 Key Findings

- **91% of smartphone price variance** can be explained by hardware specifications alone (R² = 0.91).
- **Weight** is the strongest price predictor (importance = 0.37) — heavier phones use premium materials.
- **The same phone can cost 13–83% more** depending on the country of purchase.
- **KMeans (K=4)** produces clusters that closely align with actual price tiers (Silhouette = 0.28).
