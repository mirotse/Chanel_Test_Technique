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


######## SCORE TOP 10 ################
def display_top_matches_with_scores(top_matches, similarity_matrix):
    """
    Display the top matching CVs for each job along with their similarity scores.

    :param top_matches: A dictionary where keys are job identifiers and values are arrays of top CV indices.
    :param similarity_matrix: The matrix containing similarity scores.
    """
    for job, top_indices in top_matches.items():
        print(f"Top Matches for {job} with Scores:")

        # Assuming job format is 'Job_X' where X is the numeric index
        job_index = int(job.split('_')[1]) - 1  

        for index in top_indices:
            score = similarity_matrix[job_index, index]
            print(f"CV {index}: Score = {score}")
        print()  # For a blank line between different jobs