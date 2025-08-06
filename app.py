import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Page Configuration ---
# Sets the title and icon that appear in the browser tab, and uses a wide layout.
st.set_page_config(
    page_title="Indian Dowry Case Search",
    page_icon="⚖️",
    layout="wide"
)

# --- 2. Data Loading and Model Caching ---
# @st.cache_data tells Streamlit to run this function only once, speeding up the app.
@st.cache_data
def load_data_and_model():
    """
    This function loads the dataset from the CSV and builds the TF-IDF model.
    It's cached to avoid reloading and re-calculating on every user interaction.
    """
    try:
        # Load the dataset using the correct filename we identified.
        df = pd.read_csv('dowry_cases_v1.csv.csv')
        # Ensure the 'Judgement_Text' column has no missing values.
        df['Judgement_Text'] = df['Judgement_Text'].fillna('')
        
        # Initialize and fit the TF-IDF Vectorizer.
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_matrix = vectorizer.fit_transform(df['Judgement_Text'])
        
        return df, vectorizer, tfidf_matrix
    except FileNotFoundError:
        # Display a user-friendly error on the webpage if the file is missing.
        st.error("Error: 'dowry_cases_v1.csv.csv' not found. Please make sure it's in the same directory as the app.py file.")
        return None, None, None

# Load the data and the model when the app starts.
df, vectorizer, tfidf_matrix = load_data_and_model()

# --- 3. Webpage User Interface (UI) ---
st.title("⚖️ Indian Dowry Case Law Search Engine")
st.write(
    "This tool helps you find relevant case judgements from Indian courts. "
    "Enter keywords related to your search below."
)

# --- 4. Search Functionality ---
# We proceed only if the data was loaded successfully.
if df is not None:
    # Create a text input box for the user's search query.
    user_query = st.text_input(
        "Search for keywords (e.g., 'cruelty by husband', 'section 498a', 'demand for property')",
        ""
    )

    # Only run the search if the user has typed something.
    if user_query:
        # Transform the user's query using the fitted vectorizer.
        query_vector = vectorizer.transform([user_query])
        
        # Calculate cosine similarity between the query and all judgements.
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get the indices of the top 5 most similar documents.
        top_n_indices = cosine_similarities.argsort()[-5:][::-1]
        
        st.subheader(f"Top 5 most relevant cases for: '{user_query}'")

        # --- 5. Display Results ---
        # Loop through the top results and display them.
        for i, index in enumerate(top_n_indices):
            # Use st.container to group the elements for each result.
            with st.container():
                st.markdown("---") # Visual separator
                
                # Display Rank and Similarity Score.
                st.markdown(f"**Rank:** {i+1} | **Similarity Score:** {cosine_similarities[index]:.4f}")
                
                # Use st.expander to show the full text only when the user clicks.
                # The title of the expander is the case title.
                with st.expander(f"**Title: {df.iloc[index]['Title']}**"):
                    st.write(df.iloc[index]['Judgement_Text'])

else:
    st.warning("Could not load the dataset. The app cannot function without the data file.")

