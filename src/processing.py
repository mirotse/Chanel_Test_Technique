import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import os
import docx

def select_and_save_job_descriptions(input_file, output_file, num_descriptions=300):
    """
    Reads job descriptions from a CSV file, selects a sample, and saves them to another CSV.

    Parameters:
    - input_file: String. Path to the input CSV file with job descriptions.
    - output_file: String. Path to the output CSV file to save the sample.
    - num_descriptions: Integer, optional. Number of descriptions to sample (default 300).
    """
    df = pd.read_csv(input_file)
    subset_df = df.sample(n=num_descriptions)
    
    jobpost_df = subset_df[['jobpost']]
    
    jobpost_df.to_csv(output_file, index=False)


def preprocess_text(text):
    """
    Processes text by converting to lowercase, removing non-word characters, 
    eliminating stopwords, and applying stemming.

    Parameters:
    - text: String. The input text to process.

    Returns:
    - String. The processed text.
    """
    text = text.lower()
    text = re.sub(r'\W+', ' ', text) # Remove non-word characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


def read_docx(file_path):
    """
    Reads a DOCX file and returns its content as text.

    Parameters:
    - file_path: String. The path to the DOCX file.

    Returns:
    - String. The text content of the DOCX file.
    """
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_cv_texts(cv_folder_path):
    """
    Reads all DOCX files in a given folder and returns their contents as a list of texts.

    Parameters:
    - cv_folder_path: String. The path to the folder containing DOCX files.

    Returns:
    - List of String. The text contents of each DOCX file in the folder.
    """
    cv_texts = []
    for file in os.listdir(cv_folder_path):
        if file.endswith(".docx"):
            file_path = os.path.join(cv_folder_path, file)
            text = read_docx(file_path)
            cv_texts.append(text)
    return cv_texts