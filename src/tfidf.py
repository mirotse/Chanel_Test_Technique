from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text_with_tfidf(text):
    """
    Vectorizes the input text using the TF-IDF (Term Frequency-Inverse Document Frequency) method.

    This function applies TF-IDF vectorization to the given text with unigrams and bigrams.

    Parameters:
    - text: List of String. The text to be vectorized.

    Returns:
    - Tuple. The first element is the TF-IDF matrix, and the second element is the vectorizer object.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(text)
    return tfidf_matrix, vectorizer


def get_top_matches_for_each_job(similarity_matrix, top_n=10):
    """
    Identifies the top N matches for each job in a similarity matrix.

    This function iterates over each job in the similarity matrix and finds the indices of the top N most
    similar items, assuming the rows represent different jobs and columns represent their similarity to other items.

    Parameters:
    - similarity_matrix: NumPy array. A square matrix representing similarity scores between items.
    - top_n: Integer, optional. The number of top matches to find for each job (default is 10).

    Returns:
    - Dictionary. Keys are job identifiers ('Job_1', 'Job_2', etc.), and values are lists of indices of the top matches.
    """
    top_matches = {}
    for i in range(similarity_matrix.shape[0]):  # For each job description
        sorted_indices = similarity_matrix[i].argsort()[::-1][:top_n]  # Sort and get top indices
        top_matches[f'Job_{i+1}'] = sorted_indices
    return top_matches
