"""
Spotify Insight Engine — Streamlit App
Mirrors exactly: spotify-insight-engine.ipynb

Folder structure expected:
  data/cleaned_spotify.csv   (or data/cleaned_spotify)
  models/hit_predictor.pkl
  models/scaler.pkl
  models/feature_columns.pkl
  models/kmeans.pkl          (optional — retrains if missing)

Run:
  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Insight Engine",
    page_icon="🎵",
    layout="wide",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🎵 Spotify Insight Engine")
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Dashboard",
        "🎯 Hit Predictor",
        "🔍 Recommendations",
        "👤 Artist & Genre",
        "🎧 Mood Playlist",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built from spotify-insight-engine.ipynb")

# ─── Feature lists (exact match to notebook) ──────────────────────────────────
FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_mins",
]

REC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "valence", "tempo",
]

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "acousticness",
    "tempo", "loudness", "speechiness", "instrumentalness",
]

CLUSTER_NAMES = {
    0: "Party Songs",
    1: "Workout Songs",
    2: "Calm Acoustic",
    3: "Emotional Songs",
    4: "Relaxing Songs",
}

# ─── Load cleaned data ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in ["data/cleaned_spotify.csv", "data/cleaned_spotify"]:
        if os.path.exists(path):
            return pd.read_csv(path)
    st.error(
        "❌ Dataset not found. Run the notebook first — it saves `data/cleaned_spotify.csv`."
    )
    st.stop()


df = load_data()
df = df.reset_index(drop=True)

# Add duration_mins if missing
if "duration_mins" not in df.columns and "duration_ms" in df.columns:
    df["duration_mins"] = df["duration_ms"] / 60000

# ─── Build recommendation matrix (mirrors notebook df_rec + rec_scaler) ───────
@st.cache_data
def build_rec_matrix(_df):
    df_rec = _df[REC_FEATURES + ["track_name", "artists", "track_genre", "popularity"]].reset_index(drop=True)
    rec_scaler = MinMaxScaler()
    X_rec = rec_scaler.fit_transform(df_rec[REC_FEATURES])
    return df_rec, X_rec


df_rec, X_recommended = build_rec_matrix(df)

# ─── Build cluster data (mirrors notebook: 20k sample + KMeans k=5) ───────────
@st.cache_data
def build_cluster_data(_df):
    df_cl = _df.sample(20000, random_state=42).reset_index(drop=True)
    cl_scaler = MinMaxScaler()
    X_cl = cl_scaler.fit_transform(df_cl[AUDIO_FEATURES])

    if os.path.exists("models/kmeans.pkl"):
        kmeans = joblib.load("models/kmeans.pkl")
        df_cl["cluster"] = kmeans.predict(X_cl)
    else:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df_cl["cluster"] = kmeans.fit_predict(X_cl)

    df_cl["cluster_name"] = df_cl["cluster"].map(CLUSTER_NAMES)
    return df_cl


df_cluster = build_cluster_data(df)

# ─── Load hit-prediction models ───────────────────────────────────────────────
@st.cache_resource
def load_hit_models():
    try:
        model     = joblib.load("models/hit_predictor.pkl")
        scaler    = joblib.load("models/scaler.pkl")
        feat_cols = joblib.load("models/feature_columns.pkl")
        return model, scaler, feat_cols
    except FileNotFoundError:
        return None, None, None


hit_model, hit_scaler, feat_cols = load_hit_models()

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION FUNCTIONS  (exact logic from notebook)
# ══════════════════════════════════════════════════════════════════════════════

def normal_recommendation(song_name, top_n):
    matches = df_rec[df_rec["track_name"].str.lower() == song_name.strip().lower()]
    if matches.empty:
        return None
    idx = matches.index[0]
    song_vector = X_recommended[idx].reshape(1, -1)
    sim_scores = cosine_similarity(song_vector, X_recommended)[0]
    temp_df = df_rec.copy()
    temp_df["similarity_score"] = sim_scores
    temp_df = temp_df[temp_df.index != idx]
    temp_df = temp_df[temp_df["popularity"] >= 0.65]
    temp_df = temp_df.sort_values("similarity_score", ascending=False)
    temp_df = temp_df.drop_duplicates(subset="track_name")
    result = temp_df[["track_name", "artists", "track_genre", "popularity", "similarity_score"]].head(top_n).reset_index(drop=True)
    result.index += 1
    return result


def cluster_recommendation(song_name, top_n):
    matches = df_cluster[df_cluster["track_name"].str.lower() == song_name.strip().lower()]
    if matches.empty:
        return None, None
    cluster_label = matches["cluster"].values[0]
    cluster_songs = df_cluster[df_cluster["cluster"] == cluster_label]
    cluster_songs = cluster_songs[cluster_songs["track_name"].str.lower() != song_name.strip().lower()]
    cluster_songs = cluster_songs[cluster_songs["popularity"] >= 0.65]
    cluster_songs = cluster_songs.drop_duplicates(subset="track_name")
    top_songs = cluster_songs.sort_values("popularity", ascending=False).head(top_n)[["track_name", "artists", "track_genre", "popularity"]].reset_index(drop=True)
    top_songs.index += 1
    return top_songs, cluster_label


def hybrid_recommendation(song_name, top_n, min_popularity=0.65, w_popularity=0.5, w_similarity=0.3, w_cluster=0.2):
    cosine_recs = normal_recommendation(song_name, top_n * 2)
    if cosine_recs is None:
        return None
    cluster_result = cluster_recommendation(song_name, top_n * 2)
    if cluster_result[0] is None:
        return None
    cluster_recs, _ = cluster_result
    cosine_recs  = cosine_recs.reset_index(drop=True)
    cluster_recs = cluster_recs.reset_index(drop=True)
    merged = pd.merge(cosine_recs, cluster_recs, on=["track_name", "artists", "track_genre", "popularity"], how="outer")
    merged["similarity_score"] = merged.get("similarity_score", pd.Series(0.0, index=merged.index)).fillna(0)
    merged["combined_score"] = (
        w_popularity * merged["popularity"]
        + w_similarity * merged["similarity_score"]
        + w_cluster * (merged["similarity_score"] > 0).astype(int)
    )
    filtered = merged[merged["popularity"] >= min_popularity]
    final = filtered.sort_values("combined_score", ascending=False).head(top_n)[["track_name", "artists", "track_genre", "popularity", "combined_score"]].reset_index(drop=True)
    final.index += 1
    return final


def artist_songs(artist_name, top_n):
    songs = df_rec[df_rec["artists"].str.lower().str.contains(artist_name.strip().lower(), na=False)]
    if songs.empty:
        return None
    result = songs[["track_name", "artists", "track_genre", "popularity"]].sort_values("popularity", ascending=False).head(top_n).reset_index(drop=True)
    result.index += 1
    return result


def recommend_genres(genre, top_n):
    genre_songs = df_rec[df_rec["track_genre"].str.lower() == genre.strip().lower()]
    if genre_songs.empty:
        return None
    result = genre_songs.sort_values("popularity", ascending=False).head(top_n)[["track_name", "artists", "popularity"]].reset_index(drop=True)
    result.index += 1
    return result


def mood_based(mood_type, top_n):
    rules = {
        "workout": (df["energy"] > 0.7) & (df["tempo"] > 0.6),
        "study":   (df["instrumentalness"] > 0.4) & (df["energy"] < 0.5),
        "relax":   (df["acousticness"] > 0.6) & (df["energy"] < 0.4),
        "party":   (df["danceability"] > 0.7) & (df["energy"] > 0.6),
        "happy":   (df["valence"] > 0.7) & (df["energy"] > 0.5),
        "sad":     (df["valence"] < 0.3) & (df["energy"] < 0.5),
    }
    if mood_type not in rules:
        return None
    filtered = df[rules[mood_type]]
    result = filtered.sort_values("popularity", ascending=False).drop_duplicates(subset="track_name").head(top_n)[["track_name", "artists", "popularity"]].reset_index(drop=True)
    result.index += 1
    return result


def predict_hit_probability(song_features):
    if hit_model is None:
        return None
    song_vector = np.array([song_features[f] for f in feat_cols]).reshape(1, -1)
    song_vector_scaled = hit_scaler.transform(song_vector)
    return hit_model.predict_proba(song_vector_scaled)[0][1]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎵 Total Songs",   f"{len(df):,}")
    c2.metric("🎼 Genres",        f"{df['track_genre'].nunique()}")
    c3.metric("🔥 Hit Songs",     f"{(df['popularity'] >= 0.65).sum():,}")
    c4.metric("⭐ Avg Popularity", f"{df['popularity'].mean():.3f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Popularity Distribution")
        fig = px.histogram(df, x="popularity", nbins=20, color_discrete_sequence=["#1DB954"],
                           labels={"popularity": "Popularity", "count": "Frequency"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 20 Genres by Avg Popularity")
        top_genres = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(20)
        fig = px.bar(x=top_genres.index, y=top_genres.values,
                     labels={"x": "Genre", "y": "Average Popularity"},
                     color=top_genres.values, color_continuous_scale="Viridis")
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Audio Feature Distribution")
    hist_feat = st.selectbox("Select feature", FEATURES)
    fig = px.histogram(df, x=hist_feat, nbins=30, color_discrete_sequence=["#1DB954"],
                       labels={hist_feat: hist_feat.replace("_", " ").title()})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap (Audio Features + Popularity)")
    corr_cols = [c for c in FEATURES + ["popularity"] if c in df.columns]
    corr = df[corr_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="YlGnBu", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗂️ Song Cluster Distribution (20k sample)")
    cluster_counts = df_cluster["cluster_name"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    fig = px.pie(cluster_counts, names="Cluster", values="Count", hole=0.4,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HIT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Hit Predictor":
    st.title("🎯 Hit Song Predictor")
    st.markdown("Enter audio features. The Random Forest model returns the hit probability (popularity ≥ 0.65).")
    st.markdown("---")

    if hit_model is None:
        st.error("Model files not found. Run the notebook to generate `models/hit_predictor.pkl`, `models/scaler.pkl`, `models/feature_columns.pkl`.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            danceability     = st.slider("💃 Danceability",     0.0,   1.0,  0.82, 0.01)
            energy           = st.slider("⚡ Energy",            0.0,   1.0,  0.88, 0.01)
            loudness         = st.slider("🔊 Loudness (dB)",   -60.0,  0.0, -4.5,  0.5)
            speechiness      = st.slider("🗣️ Speechiness",      0.0,   1.0,  0.05, 0.01)
            acousticness     = st.slider("🎸 Acousticness",     0.0,   1.0,  0.10, 0.01)
        with col2:
            instrumentalness = st.slider("🎹 Instrumentalness", 0.0,   1.0,  0.00, 0.01)
            liveness         = st.slider("🎤 Liveness",          0.0,   1.0,  0.15, 0.01)
            valence          = st.slider("😊 Valence",           0.0,   1.0,  0.80, 0.01)
            tempo            = st.slider("🥁 Tempo (BPM)",      50.0, 220.0, 128.0, 1.0)
            duration_mins    = st.slider("⏱️ Duration (mins)",   0.5,   8.0,  3.5,  0.1)

        if st.button("🎵 Predict Hit Probability", type="primary"):
            song_features = {
                "danceability": danceability, "energy": energy, "loudness": loudness,
                "speechiness": speechiness, "acousticness": acousticness,
                "instrumentalness": instrumentalness, "liveness": liveness,
                "valence": valence, "tempo": tempo, "duration_mins": duration_mins,
            }
            prob  = predict_hit_probability(song_features)
            label = "🔥 HIT" if prob >= 0.3 else "❌ Not a Hit"

            st.markdown("---")
            m1, m2 = st.columns(2)
            m1.metric("Prediction",      label)
            m2.metric("Hit Probability", f"{prob:.2%}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                title={"text": "Hit Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#1DB954" if prob >= 0.3 else "#e63946"},
                    "steps": [
                        {"range": [0,  50], "color": "#ffddd2"},
                        {"range": [50, 100], "color": "#d4edda"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": 50},
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            reasons = []
            if energy > 0.7:        reasons.append("high energy")
            if danceability > 0.7:  reasons.append("high danceability")
            if valence > 0.6:       reasons.append("positive mood")
            if loudness > -5:       reasons.append("loud production")
            if tempo > 120:         reasons.append("fast tempo")
            reason_str = ", ".join(reasons) if reasons else "moderate audio profile"
            st.info(f"💡 Predicted as **{label}** because of: **{reason_str}**.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Recommendations":
    st.title("🔍 Song Recommendations")
    st.markdown("---")

    mode    = st.radio("Mode", ["🔵 Normal (Cosine Similarity)", "🟢 Cluster-Based", "🟡 Hybrid"], horizontal=True)
    song_in = st.text_input("🎵 Song name", placeholder="e.g. dream on")
    top_n   = st.slider("Results", 5, 20, 10)

    if mode == "🟡 Hybrid":
        st.markdown("**Hybrid Weights** (sum to 1.0)")
        wc1, wc2, wc3 = st.columns(3)
        w_pop = wc1.number_input("Popularity weight", 0.0, 1.0, 0.5, 0.1)
        w_sim = wc2.number_input("Similarity weight", 0.0, 1.0, 0.3, 0.1)
        w_cls = wc3.number_input("Cluster weight",    0.0, 1.0, 0.2, 0.1)
        if abs(round(w_pop + w_sim + w_cls, 2) - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {round(w_pop+w_sim+w_cls,2)} — should sum to 1.0")

    if st.button("Find Similar Songs", type="primary") and song_in.strip():
        with st.spinner("Searching..."):
            if mode == "🔵 Normal (Cosine Similarity)":
                result = normal_recommendation(song_in, top_n)
                if result is None:
                    st.error(f'"{song_in}" not found.')
                else:
                    st.success(f"✅ {len(result)} songs similar to **{song_in}** — Normal")
                    st.dataframe(result, use_container_width=True)
                    fig = px.bar(result, x="similarity_score", y="track_name", orientation="h",
                                 color="similarity_score", color_continuous_scale="Greens",
                                 title="Similarity Scores", labels={"track_name": "Song"})
                    fig.update_layout(yaxis={"autorange": "reversed"})
                    st.plotly_chart(fig, use_container_width=True)

            elif mode == "🟢 Cluster-Based":
                result, cluster_label = cluster_recommendation(song_in, top_n)
                if result is None:
                    st.error(f'"{song_in}" not found in cluster dataset.')
                else:
                    cname = CLUSTER_NAMES.get(cluster_label, f"Cluster {cluster_label}")
                    st.success(f"✅ {len(result)} songs from cluster **{cname}**")
                    st.dataframe(result, use_container_width=True)
                    fig = px.bar(result, x="popularity", y="track_name", orientation="h",
                                 color="popularity", color_continuous_scale="Viridis",
                                 title=f"Popularity — {cname}", labels={"track_name": "Song"})
                    fig.update_layout(yaxis={"autorange": "reversed"})
                    st.plotly_chart(fig, use_container_width=True)

            else:
                result = hybrid_recommendation(song_in, top_n, w_popularity=w_pop, w_similarity=w_sim, w_cluster=w_cls)
                if result is None:
                    st.error(f'"{song_in}" not found.')
                else:
                    st.success(f"✅ {len(result)} Hybrid recommendations for **{song_in}**")
                    st.dataframe(result, use_container_width=True)
                    fig = px.bar(result, x="combined_score", y="track_name", orientation="h",
                                 color="combined_score", color_continuous_scale="YlGnBu",
                                 title="Combined Score", labels={"track_name": "Song"})
                    fig.update_layout(yaxis={"autorange": "reversed"})
                    st.plotly_chart(fig, use_container_width=True)

    # Side-by-side comparison
    st.markdown("---")
    st.subheader("📊 Compare All 3 Systems")
    cmp_song = st.text_input("Song for comparison", placeholder="e.g. dream on", key="cmp")
    cmp_n    = st.slider("Results per system", 3, 10, 5, key="cmpn")

    if st.button("Compare", type="secondary") and cmp_song.strip():
        nr = normal_recommendation(cmp_song, cmp_n)
        cr, cr_lbl = cluster_recommendation(cmp_song, cmp_n)
        hr = hybrid_recommendation(cmp_song, cmp_n)
        if nr is None:
            st.error(f'"{cmp_song}" not found.')
        else:
            t1, t2, t3 = st.columns(3)
            with t1:
                st.markdown("**🔵 Normal**")
                st.dataframe(nr[["track_name", "similarity_score"]], use_container_width=True)
            with t2:
                st.markdown("**🟢 Cluster-Based**")
                if cr is not None:
                    st.dataframe(cr[["track_name", "popularity"]], use_container_width=True)
                else:
                    st.warning("Not in cluster dataset.")
            with t3:
                st.markdown("**🟡 Hybrid**")
                if hr is not None:
                    st.dataframe(hr[["track_name", "combined_score"]], use_container_width=True)
                else:
                    st.warning("Could not compute.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ARTIST & GENRE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Artist & Genre":
    st.title("👤 Artist & Genre Explorer")
    st.markdown("---")

    st.subheader("🎤 Top Songs by Artist")
    artist_in = st.text_input("Artist name", placeholder="e.g. michael jackson")
    artist_n  = st.slider("Songs to show", 5, 20, 10, key="an")

    if st.button("Search Artist", type="primary") and artist_in.strip():
        result = artist_songs(artist_in, artist_n)
        if result is None:
            st.error(f'No songs found for "{artist_in}".')
        else:
            st.success(f"✅ Top {len(result)} songs by **{artist_in}**")
            st.dataframe(result, use_container_width=True)
            fig = px.bar(result, x="popularity", y="track_name", orientation="h",
                         color="popularity", color_continuous_scale="Blues",
                         title=f"Popularity — {artist_in}", labels={"track_name": "Song"})
            fig.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🎼 Top Songs by Genre")
    all_genres = sorted(df["track_genre"].dropna().unique())
    genre_sel  = st.selectbox("Genre", all_genres)
    genre_n    = st.slider("Songs to show", 5, 30, 20, key="gn")

    if st.button("Browse Genre", type="primary"):
        result = recommend_genres(genre_sel, genre_n)
        if result is None:
            st.error(f'Genre "{genre_sel}" not found.')
        else:
            st.success(f"✅ Top {len(result)} songs in **{genre_sel}**")
            st.dataframe(result, use_container_width=True)
            fig = px.bar(result, x="popularity", y="track_name", orientation="h",
                         color="popularity", color_continuous_scale="Purples",
                         title=f"Top Songs — {genre_sel}", labels={"track_name": "Song"})
            fig.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MOOD PLAYLIST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎧 Mood Playlist":
    st.title("🎧 Mood Playlist Generator")
    st.markdown("---")

    mood_icons = {"workout": "🏋️", "study": "📚", "relax": "😌",
                  "party": "🎉", "happy": "😄", "sad": "😢"}

    mood_sel = st.selectbox("Choose your mood", list(mood_icons.keys()),
                             format_func=lambda m: f"{mood_icons[m]} {m.capitalize()}")
    mood_n   = st.slider("Playlist size", 5, 30, 10)

    if st.button("🎵 Generate Playlist", type="primary"):
        result = mood_based(mood_sel, mood_n)
        if result is None or result.empty:
            st.warning("No songs found for this mood.")
        else:
            st.success(f"✅ {len(result)}-song **{mood_icons[mood_sel]} {mood_sel.capitalize()}** playlist")
            st.dataframe(result, use_container_width=True)
            fig = px.bar(result, x="popularity", y="track_name", orientation="h",
                         color="popularity", color_continuous_scale="RdYlGn",
                         title=f"{mood_icons[mood_sel]} {mood_sel.capitalize()} Playlist",
                         labels={"track_name": "Song", "popularity": "Popularity"})
            fig.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ How moods are defined (from notebook)"):
        st.markdown("""
| Mood | Rule |
|---|---|
| 🏋️ Workout | energy > 0.7 AND tempo > 0.6 |
| 📚 Study | instrumentalness > 0.4 AND energy < 0.5 |
| 😌 Relax | acousticness > 0.6 AND energy < 0.4 |
| 🎉 Party | danceability > 0.7 AND energy > 0.6 |
| 😄 Happy | valence > 0.7 AND energy > 0.5 |
| 😢 Sad | valence < 0.3 AND energy < 0.5 |
        """)