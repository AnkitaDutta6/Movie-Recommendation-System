# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Load data
@st.cache_data
def load_data():
    ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    movies = pd.read_csv(
        "ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        names=["item_id", "title", "release_date", "video_release_date", "IMDb_URL"] + [f"genre_{i}" for i in range(19)]
    )
    return ratings, movies

ratings_df, movies_df = load_data()

# Preprocessing
movies_df['genres'] = movies_df.iloc[:, 5:].apply(
    lambda row: ' '.join([movies_df.columns[5+i] for i, val in enumerate(row) if val == 1]), axis=1)
movie_data = pd.merge(ratings_df, movies_df[["item_id", "title"]], on="item_id")

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# SVD Collaborative Filtering
user_movie_matrix = movie_data.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(user_movie_matrix)
movie_factors = svd.components_.T
collab_sim = cosine_similarity(movie_factors, movie_factors)
collab_indices = pd.Series(data=np.arange(len(user_movie_matrix.columns)), index=user_movie_matrix.columns)

# Hybrid recommender
def get_hybrid_recommendations(title, alpha=0.5, top_n=10):
    if title not in indices or title not in collab_indices:
        return []
    idx_content = indices[title]
    idx_collab = collab_indices[title]
    shared_titles = list(set(indices.index) & set(collab_indices.index))
    shared_titles = [t for t in shared_titles if t != title]
    scores = []
    for t in shared_titles:
        try:
            idx_c = indices[t]
            idx_cf = collab_indices[t]
            content_sim = float(cosine_sim[idx_content][idx_c])
            collab_sim_score = float(collab_sim[idx_collab][idx_cf])
            hybrid_score = alpha * content_sim + (1 - alpha) * collab_sim_score
            scores.append((t, content_sim, collab_sim_score, hybrid_score))
        except:
            continue
    scores = sorted(scores, key=lambda x: x[3], reverse=True)
    return scores[:top_n]

# Streamlit UI
st.title("🎬 Hybrid Movie Recommender System")

movie_list = sorted(list(set(indices.index) & set(collab_indices.index)))
selected_movie = st.selectbox("Select a movie:", movie_list)
alpha = st.slider("Balance (α): Content-Based vs Collaborative", 0.0, 1.0, 0.5, step=0.05)
top_n = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Get Recommendations"):
    recs = get_hybrid_recommendations(selected_movie, alpha=alpha, top_n=top_n)
    if not recs:
        st.warning("No recommendations found. Try another title.")
    else:
        titles, content_scores, collab_scores, hybrid_scores = zip(*recs)
        rec_df = pd.DataFrame({
            "Title": titles,
            "Content-Based Score": content_scores,
            "Collaborative Score": collab_scores,
            "Hybrid Score": hybrid_scores
        })
        st.dataframe(rec_df)

        # Plot
        st.subheader("📊 Recommendation Score Breakdown")
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(titles))
        width = 0.25
        ax.bar(x - width, content_scores, width, label='Content')
        ax.bar(x, collab_scores, width, label='Collaborative')
        ax.bar(x + width, hybrid_scores, width, label='Hybrid')
        ax.set_xticks(x)
        ax.set_xticklabels(titles, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title(f"Top {top_n} Recommendations for '{selected_movie}'")
        ax.legend()
        st.pyplot(fig)