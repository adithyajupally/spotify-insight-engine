# 🎵 Spotify Insight Engine

> End-to-End Data Science & Machine Learning project using the Spotify Tracks Dataset.


## Live App: https://spotify-insight-engine-tfbyt2nupntmfsoniwhxzl.streamlit.app/


## 🚀 Project Overview

This project analyzes 114,000+ Spotify songs to:
- **Predict** whether a song becomes a hit using audio features
- **Recommend** similar songs using content-based filtering
- **Cluster** songs into hidden music groups
- **Visualize** music trends through an interactive Streamlit dashboard

---

## 📁 Project Structure

```
spotify-insight-engine/
│
├── data/
│   ├── dataset.csv              ← Download from Kaggle
│   └── cleaned_spotify.csv      ← Generated after running notebook
│
├── models/
│   ├── hit_predictor.pkl
│   └── scaler.pkl
│
├── visuals/
│   ├── popularity_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── mood_map.html
│   ├── pca_clusters.html
│   └── tsne_clusters.html
│
├── app/
│   └── streamlit_app.py
│
├── spotify-insight-engine.ipynb
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source:** [Spotify Tracks Dataset – Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

| Info | Value |
|------|-------|
| Songs | 114,000+ |
| Genres | 125+ |
| Features | popularity, danceability, energy, valence, tempo, etc. |

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/spotify-intelligence-platform.git
cd spotify-intelligence-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place it in data/dataset.csv

# 4. Run the notebook (Jupyter)
jupyter notebook spotify-insight-engine.ipynb

# 5. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

---

## 🔬 Phases

### Phase 1 — Data Cleaning
- Removed duplicates on `track_name + artists`
- Handled missing values
- Engineered features: `mood_score`, `energy_loudness_ratio`, `duration_minutes`, `normalized_popularity`
- Scaled audio features with MinMaxScaler

### Phase 2 — Exploratory Data Analysis
- Popularity distribution, genre popularity, tempo distribution
- Correlation heatmap between audio features
- Mood map using valence and energy (Happy, Sad, Energetic, Calm)
- Interactive Plotly charts

### Phase 3 — Hit Song Prediction
- Binary target: `hit = 1` if popularity ≥ 75
- Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM
- Best model: **XGBoost**
- Explainable AI: SHAP summary plots

### Phase 4 — Music Recommendation System
- Content-based filtering using cosine similarity on audio features
- Mood-based playlists: Workout, Study, Relax, Party, Happy, Sad
- Genre-based top song recommendations

### Phase 5 — Song Clustering
- KMeans (K=5): Party, Workout, Calm Acoustic, Emotional, Relaxing
- Dimensionality reduction: PCA and t-SNE for 2D visualization

### Phase 6 — Streamlit App
Four interactive pages:
- 📊 Dashboard
- 🎯 Hit Predictor
- 🔍 Recommendations
- 🎧 Mood Playlist Generator

---

## 🤖 ML Model Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.86 | ~0.42 | ~0.78 |
| Decision Tree | ~0.87 | ~0.45 | ~0.72 |
| Random Forest | ~0.89 | ~0.52 | ~0.85 |
| Gradient Boosting | ~0.90 | ~0.55 | ~0.87 |
| XGBoost | ~0.91 | ~0.57 | ~0.89 |
| LightGBM | ~0.91 | ~0.56 | ~0.88 |

*(Actual values will vary based on your data split)*

---

## 🔮 Future Improvements

- Spotify API integration for real-time data
- NLP analysis on song lyrics
- Deep learning recommendation system
- Personalized user recommendations
- Playlist export feature

---

## 💼 Resume Description

> *"Developed a Spotify Intelligence Platform using Machine Learning and audio feature analysis to predict hit songs, generate music recommendations, cluster songs into hidden categories, and visualize music trends through an interactive Streamlit application."*
