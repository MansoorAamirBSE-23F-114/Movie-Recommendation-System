import streamlit as st

from recommender import build_model, download_data, recommend

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading data and building the model (first run only)...")
def load():
    download_data()
    return build_model()


movies, content, collab = load()

st.title("🎬 Movie Recommender")
st.caption("Pick movies you enjoyed and get similar suggestions (MovieLens data).")

liked = st.multiselect("Movies you like", movies["title"].tolist(), max_selections=10)
n = st.slider("Number of recommendations", 5, 20, 10)
cf_weight = st.slider("Weight on 'viewers also liked' vs genre similarity", 0.0, 1.0, 0.7)
min_ratings = st.slider("Minimum number of ratings a movie must have", 1, 100, 10)

if st.button("Recommend", type="primary"):
    if not liked:
        st.warning("Please pick at least one movie.")
    else:
        recs = recommend(movies, content, collab, liked, n, cf_weight, min_ratings)
        if recs.empty:
            st.info("No matches found. Try lowering the minimum ratings.")
        else:
            st.dataframe(recs, use_container_width=True, hide_index=True)
