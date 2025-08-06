import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Indian Dowry Case Search",
    page_icon="⚖️",
    layout="wide"
)

# --- 2. Data Loading and Model Caching ---
@st.cache_data
def load_data_and_model():
    """
    Loads the dataset and builds the TF-IDF model.
    Cached to avoid reloading on every interaction.
    """
    try:
        df = pd.read_csv('dowry_cases_v1.csv.csv')
        df['Judgement_Text'] = df['Judgement_Text'].fillna('')
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_matrix = vectorizer.fit_transform(df['Judgement_Text'])
        return df, vectorizer, tfidf_matrix
    except FileNotFoundError:
        st.error("Error: 'dowry_cases_v1.csv.csv' not found. Please make sure it's in the same directory.")
        return None, None, None

# --- 3. Session State Initialization ---
# Initialize session_state to store search history.
# This runs only once per user session.
if 'history' not in st.session_state:
    st.session_state.history = []


# Load the data and the model.
df, vectorizer, tfidf_matrix = load_data_and_model()


# --- 4. Sidebar for Search History ---
with st.sidebar:
    st.header("Search History")
    # Display the history if it's not empty, showing the most recent searches first.
    if st.session_state.history:
        for query in reversed(st.session_state.history):
            st.info(query)
    else:
        st.write("No searches have been made yet.")


# --- 5. Main Page UI ---
st.title("⚖️ Indian Dowry Case Law Search Engine")
st.write(
    "This tool helps you find relevant case judgements from Indian courts. "
    "Enter keywords related to your search below."
)

# --- 6. Search Functionality ---
if df is not None:
    user_query = st.text_input(
        "Search for keywords (e.g., 'cruelty by husband', 'section 498a', 'demand for property')",
        ""
    )

    if user_query:
        # --- Add query to history ---
        # Add the current query to the history list if it's not already the most recent one.
        if not st.session_state.history or st.session_state.history[-1] != user_query:
            st.session_state.history.append(user_query)

        # --- Perform Search ---
        query_vector = vectorizer.transform([user_query])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_n_indices = cosine_similarities.argsort()[-5:][::-1]
        
        st.subheader(f"Top 5 most relevant cases for: '{user_query}'")

        # --- 7. Display Results ---
        for i, index in enumerate(top_n_indices):
            with st.container():
                st.markdown("---")
                st.markdown(f"**Rank:** {i+1} | **Similarity Score:** {cosine_similarities[index]:.4f}")
                with st.expander(f"**Title: {df.iloc[index]['Title']}**"):
                    st.write(df.iloc[index]['Judgement_Text'])
else:
    st.warning("Could not load the dataset. The app cannot function without the data file.")


