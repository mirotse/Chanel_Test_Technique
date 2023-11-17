from processing import *
from tfidf import *
from sklearn.metrics.pairwise import cosine_similarity


# Set the paths for input and output files
input_file = '/home/rooting/code/Chanel/Job posts/data job posts.csv'
output_file = '/home/rooting/code/Chanel/Job posts/subset_data_job.csv'

# Select and save job descriptions from the input file to the output file
df_test = select_and_save_job_descriptions(input_file, output_file)

# Apply text preprocessing to the 'jobpost' column of the DataFrame
df_test['processed'] = df_test['jobpost'].apply(preprocess_text)

# Set the path to the folder containing CVs
cv_folder_path = '/home/rooting/code/Chanel/Resumes'

# Read all CV texts from the given folder path
cv_texts = read_cv_texts(cv_folder_path)

# Preprocess each CV text
preprocessed_cvs = [preprocess_text(cv) for cv in cv_texts]

# Create a DataFrame with original and processed CV texts
cv_df = pd.DataFrame({'original_cv': cv_texts, 'processed_cv': preprocessed_cvs})

# Combine processed job descriptions and CVs into a single list
combined_texts = df_test['processed'].tolist() + cv_df['processed_cv'].tolist()

# Vectorize the combined texts using TF-IDF
tfidf_combined_matrix, vectorizer = vectorize_text_with_tfidf(combined_texts)

# Separate the TF-IDF matrices for job posts and CVs
num_jobs = df_test.shape[0]
tfidf_matrix = tfidf_combined_matrix[:num_jobs]  # Matrix for job descriptions
tfidf_matrix_cv = tfidf_combined_matrix[num_jobs:]  # Matrix for CVs


# cosine similarity : tfidf_matrix = job descriptions / tfidf_matrix_cv = CVs
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix_cv)

top_matches = get_top_matches_for_each_job(similarity_matrix)