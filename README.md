# 🎬 Hybrid Movie Recommendation System

This project builds a **hybrid movie recommender** that combines **content-based filtering** (using movie genres) and **collaborative filtering** (via SVD). It includes a clean Streamlit app for real-time interaction and a separate statistical analysis script.

---

## 🚀 Features

* **Hybrid Recommendations**: Combines genre similarity and user rating patterns
* **Streamlit App**: Select a movie, adjust weighting, and get personalized suggestions
* **Visualization**: Bar charts for content, collaborative, and hybrid scores
* **Modular Code**: Separated logic for clarity and reuse

---

## 📊 Optional Statistical Analysis

Included in `movie_stats_analysis.py`:

* Rating distributions
* Most-rated and best-rated movies
* Popularity vs. average rating correlation

> ⚠️ These are excluded from the app for simplicity, but useful for deeper insight or interviews.

---

## 🧪 Try It Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

---

## 📌 Dataset

MovieLens 100k dataset by [GroupLens](https://grouplens.org/datasets/movielens/100k/)

---

## 💡 Future Ideas

* Deploy to Streamlit Cloud
* Add poster images via TMDb API
* Include user login + rating history
* Collaborative model using Neural MF

---

## 🧠 Author

*Built with Python, pandas, scikit-learn, and Streamlit.*
