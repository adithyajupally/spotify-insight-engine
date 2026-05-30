"""
app/streamlit_app.py
--------------------
Spotify Intelligence System — interactive Streamlit frontend.

Run with:
    streamlit run app/streamlit_app.py
from the project root directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when running from app/ or project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import streamlit as st

from src.prediction import predict_hit_probability
from src.preprocessing import (
    AUDIO_FEATURES,
    CLUSTER_FEATURES,
    REC_FEATURES,
    build_cluster_matrix,
    build_rec_matrix,
    clean_data,
    engineer_features,
    load_raw_data,
)
from src.recommenders import (
    artist_songs,
    cluster_recommendation,
    hybrid_recommendation,
    mood_based,
    normal_recommendation,
    recommend_genres,
    compare_all_recommenders,
)
from src.clustering import fit_kmeans, DEFAULT_CLUSTER_NAMES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Spotify Intelligence System",
    page_icon="🎵",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data loading so the app only preprocesses once per session
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading and preprocessing dataset…")
def load_data():
    raw = load_raw_data("data/dataset.csv")
    cleaned = clean_data(raw)
    df, _le = engineer_features(cleaned)

    df_rec, X_rec, _rec_scaler = build_rec_matrix(df)
    df_cluster_raw, X_cluster, _cluster_scaler = build_cluster_matrix(df)

    # Use saved K-Means if available, otherwise fit fresh
    try:
        kmeans = joblib.load("models/kmeans.pkl")
        df_cluster = df_cluster_raw.copy()
        df_cluster["cluster"] = kmeans.predict(X_cluster)
        df_cluster["cluster_name"] = df_cluster["cluster"].map(DEFAULT_CLUSTER_NAMES)
    except FileNotFoundError:
        df_cluster, _kmeans = fit_kmeans(df_cluster_raw, X_cluster)

    return df, df_rec, X_rec, df_cluster


df, df_rec, X_rec, df_cluster = load_data()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🎵 Spotify Intelligence")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🎯 Hit Predictor",
        "🔍 Song Recommender",
        "🎭 Mood Playlist",
        "🎤 Artist Lookup",
        "🎸 Genre Lookup",
        "🔬 Compare Recommenders",
    ],
)

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

if page == "🏠 Home":
    st.title("🎵 Spotify Intelligence System")
    st.markdown(
        """
        Welcome! This app exposes all ML features built in the Spotify
        Intelligence System notebook.

        | Page | What it does |
        |---|---|
        | **Hit Predictor** | Estimate the hit probability of a custom track |
        | **Song Recommender** | Content-based, cluster-based, or hybrid picks |
        | **Mood Playlist** | Songs matched to a mood (workout, study, party …) |
        | **Artist Lookup** | Top tracks for any artist in the dataset |
        | **Genre Lookup** | Most popular songs in a genre |
        | **Compare Recommenders** | Side-by-side comparison of all three engines |
        """
    )
    st.info(f"Dataset: **{len(df):,} tracks** loaded.")

# ---------------------------------------------------------------------------
# Hit Predictor
# ---------------------------------------------------------------------------

elif page == "🎯 Hit Predictor":
    st.header("🎯 Hit Song Predictor")
    st.markdown("Adjust the sliders to describe your track and see its predicted hit probability.")

    col1, col2 = st.columns(2)
    with col1:
        danceability    = st.slider("Danceability",     0.0, 1.0, 0.7, 0.01)
        energy          = st.slider("Energy",           0.0, 1.0, 0.8, 0.01)
        loudness        = st.slider("Loudness (dB)",   -60.0, 0.0, -5.0, 0.1)
        speechiness     = st.slider("Speechiness",      0.0, 1.0, 0.06, 0.01)
        acousticness    = st.slider("Acousticness",     0.0, 1.0, 0.1, 0.01)
    with col2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
        liveness        = st.slider("Liveness",         0.0, 1.0, 0.15, 0.01)
        valence         = st.slider("Valence",          0.0, 1.0, 0.7, 0.01)
        tempo           = st.slider("Tempo (BPM)",      50.0, 220.0, 120.0, 1.0)
        duration_mins   = st.slider("Duration (mins)",  1.0, 10.0, 3.5, 0.1)

    if st.button("Predict Hit Probability", type="primary"):
        try:
            song = {
                "danceability": danceability,
                "energy": energy,
                "loudness": loudness,
                "speechiness": speechiness,
                "acousticness": acousticness,
                "instrumentalness": instrumentalness,
                "liveness": liveness,
                "valence": valence,
                "tempo": tempo,
                "duration_mins": duration_mins,
            }
            prob = predict_hit_probability(song)
            st.metric("Hit Probability", f"{prob:.1%}")
            if prob >= 0.65:
                st.success("🔥 This track has strong hit potential!")
            elif prob >= 0.4:
                st.warning("🎵 Moderate hit potential.")
            else:
                st.error("📉 Low hit potential with current settings.")
        except FileNotFoundError:
            st.error("Model artefacts not found. Run the notebook to train and save the model first.")

# ---------------------------------------------------------------------------
# Song Recommender
# ---------------------------------------------------------------------------

elif page == "🔍 Song Recommender":
    st.header("🔍 Song Recommender")
    song_input = st.text_input("Enter a song name", placeholder="e.g. Thriller")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    rec_type = st.radio("Recommender type", ["Content-Based", "Cluster-Based", "Hybrid"])

    if st.button("Get Recommendations", type="primary") and song_input:
        with st.spinner("Finding recommendations…"):
            if rec_type == "Content-Based":
                result = normal_recommendation(song_input, df_rec, X_rec, top_n)
            elif rec_type == "Cluster-Based":
                result = cluster_recommendation(song_input, df_cluster, top_n)
            else:
                result = hybrid_recommendation(song_input, df_rec, X_rec, df_cluster, top_n)

        if result is not None and not result.empty:
            st.dataframe(result, use_container_width=True)
        else:
            st.warning(f'No recommendations found for "{song_input}".')

# ---------------------------------------------------------------------------
# Mood Playlist
# ---------------------------------------------------------------------------

elif page == "🎭 Mood Playlist":
    st.header("🎭 Mood-Based Playlist")
    mood = st.selectbox("Choose a mood", ["workout", "study", "relax", "party", "happy", "sad"])
    top_n = st.slider("Number of songs", 5, 30, 10)

    if st.button("Generate Playlist", type="primary"):
        result = mood_based(mood, df, top_n)
        if result is not None and not result.empty:
            st.subheader(f"🎵 Your {mood.capitalize()} Playlist")
            st.dataframe(result, use_container_width=True)
        else:
            st.warning("No songs found for this mood.")

# ---------------------------------------------------------------------------
# Artist Lookup
# ---------------------------------------------------------------------------

elif page == "🎤 Artist Lookup":
    st.header("🎤 Artist Lookup")
    artist_input = st.text_input("Enter an artist name", placeholder="e.g. Michael Jackson")
    top_n = st.slider("Number of tracks", 5, 30, 10)

    if st.button("Find Songs", type="primary") and artist_input:
        result = artist_songs(artist_input, df_rec, top_n)
        if result is not None and not result.empty:
            st.dataframe(result, use_container_width=True)
        else:
            st.warning(f'Artist "{artist_input}" not found in the dataset.')

# ---------------------------------------------------------------------------
# Genre Lookup
# ---------------------------------------------------------------------------

elif page == "🎸 Genre Lookup":
    st.header("🎸 Genre Lookup")
    available_genres = sorted(df["track_genre"].unique())
    genre = st.selectbox("Select a genre", available_genres)
    top_n = st.slider("Number of songs", 5, 50, 20)

    if st.button("Browse Genre", type="primary"):
        result = recommend_genres(genre, df_rec, top_n)
        if result is not None and not result.empty:
            st.dataframe(result, use_container_width=True)

# ---------------------------------------------------------------------------
# Compare Recommenders
# ---------------------------------------------------------------------------

elif page == "🔬 Compare Recommenders":
    st.header("🔬 Compare All Recommenders")
    song_input = st.text_input("Enter a song name", placeholder="e.g. Dream On")
    top_n = st.slider("Results per recommender", 3, 15, 5)

    if st.button("Compare", type="primary") and song_input:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Content-Based")
            result = normal_recommendation(song_input, df_rec, X_rec, top_n)
            if result is not None:
                st.dataframe(result[["track_name", "artists", "similarity_score"]], use_container_width=True)
            else:
                st.warning("Not found.")

        with col2:
            st.subheader("Cluster-Based")
            result = cluster_recommendation(song_input, df_cluster, top_n)
            if result is not None:
                st.dataframe(result[["track_name", "artists", "popularity"]], use_container_width=True)
            else:
                st.warning("Not found.")

        with col3:
            st.subheader("Hybrid")
            result = hybrid_recommendation(song_input, df_rec, X_rec, df_cluster, top_n)
            if result is not None:
                st.dataframe(result[["track_name", "artists", "combined_score"]], use_container_width=True)
            else:
                st.warning("Not found.")