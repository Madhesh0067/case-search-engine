import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def search_judgements(query, vectorizer, tfidf_matrix, df):
    """
    Searches for relevant judgements based on a query.
    """
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_n_indices = cosine_similarities.argsort()[-5:][::-1]
    top_n_scores = cosine_similarities[top_n_indices]
    ranked_cases = df.iloc[top_n_indices].copy()
    ranked_cases['similarity_score'] = top_n_scores
    return ranked_cases

if __name__ == "__main__":
    # Load the dataset with the correct filename
    try:
        df = pd.read_csv('dowry_cases_v1.csv.csv')
    except FileNotFoundError:
        print("Error: 'dowry_cases_v1.csv.csv' not found.")
        print("Please make sure the CSV file is in the same directory as the script.")
        exit()

    # Fill any missing values in 'Judgement_Text' with an empty string
    # This column name is correct in your file
    df['Judgement_Text'] = df['Judgement_Text'].fillna('')

    # Initialize and fit the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Judgement_Text'])

    print("TF-IDF model built successfully.")
    print("You can now search for cases. Type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your search query: ")

        if user_query.lower() == 'exit':
            break

        top_cases = search_judgements(user_query, vectorizer, tfidf_matrix, df)

        if not top_cases.empty:
            print(f"\nTop 5 most relevant cases for the query: '{user_query}'")
            rank_counter = 1
            for index, row in top_cases.iterrows():
                print("\n" + "="*50)
                print(f"Rank: {rank_counter}")
                # CORRECTED LINE: Using 'Title' instead of 'Case_ID'
                print(f"Title: {row['Title']}")
                print(f"Similarity Score: {row['similarity_score']:.4f}")
                print("\n--- Judgement Text Snippet ---")
                print(str(row['Judgement_Text'])[:500] + "...")
                print("="*50)
                rank_counter += 1
        else:
            print("No relevant cases found.")