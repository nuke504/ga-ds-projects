import pandas as pd
import numpy as np

from typing import List, Dict, Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

CATEGORICAL_COLUMNS = [
    "division",
    "region",
    "brewery_city"
]

STRING_COLUMNS_TFIDF = [
    "name",
    "style",
    "brewery_name"
]

STRING_COLUMNS_COUNT = [
    "style",
]

NUMERICAL_COLUMNS = [
    "ounces"
]

LABEL = "beer_category"

def generate_features(
    df: pd.DataFrame,
    categorical_columns: List[str] = CATEGORICAL_COLUMNS,
    string_columns_tfidf: List[str] = STRING_COLUMNS_TFIDF,
    string_columns_count: List[str] = STRING_COLUMNS_COUNT,
    numerical_columns: List[str] = NUMERICAL_COLUMNS,
    label: str = LABEL,
    svd_elements: int = 3
) -> Tuple[np.array, np.array, List[str], dict]:
    """
    Generate features from the dataframe

    - categorical columns: convert into scipy sparse matrices
    - string columns: convert into TF-IDF OR count vectoriser + truncated SVD of 3 elements
    - numerical columns: convert into numpy array
    - label: extract as seperate numpy array

    """
    features_dict = {}
    encoder_dict = {}
    column_names = {}

    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')

    def tokenise_sentence(sentence:str) -> List[str]:
        """
        Function to tokenizer sentences
        """
        word_tokens = tokenizer.tokenize(sentence)
        return [w for w in word_tokens if not w.lower() in stop_words]

    for col in categorical_columns:
        ohe = OneHotEncoder(drop="first")

        input_arr = np.array(df[col].tolist()).reshape(-1,1)

        features_dict[col] = ohe.fit_transform(input_arr).toarray()
        encoder_dict[col] = ohe

        categories = ohe.categories_[0][1:]
        categories = f"{col}_" + pd.Series(categories).str.lower().str.replace(" ","_")
        column_names[col] = categories.tolist()

    for col in string_columns_tfidf:
        tfidf = TfidfVectorizer(analyzer="word", tokenizer=tokenise_sentence)
        latent_semantic_analysis_transformer = TruncatedSVD(svd_elements, algorithm = 'arpack')

        corpus = df[col].tolist()
        document_term_matrix = tfidf.fit_transform(corpus)
        dtm_lsa = latent_semantic_analysis_transformer.fit_transform(document_term_matrix)

        features_dict[col] = dtm_lsa
        encoder_dict[col] = {"tfidf": tfidf, "lsa": latent_semantic_analysis_transformer}
        column_names[col] = [f"{col}_{i+1}" for i in range(dtm_lsa.shape[1])]

    for col in string_columns_count:
        ctv = CountVectorizer(analyzer="word", tokenizer=tokenise_sentence)
        latent_semantic_analysis_transformer = TruncatedSVD(svd_elements, algorithm = 'arpack')

        corpus = df[col].tolist()
        document_term_matrix = ctv.fit_transform(corpus).toarray().astype(np.float32)
        dtm_lsa = latent_semantic_analysis_transformer.fit_transform(document_term_matrix)

        features_dict[col] = dtm_lsa
        encoder_dict[col] = {"ctv": ctv, "lsa": latent_semantic_analysis_transformer}
        column_names[col] = [f"{col}_{i+1}" for i in range(dtm_lsa.shape[1])]

    for col in numerical_columns:
        features_dict[col] = df[col].to_numpy().reshape(-1, 1)
        column_names[col] = col

    y = df[label].to_numpy()

    X = []
    column_names_list = []
    for col, arr in features_dict.items():
        X.append(arr)
        
        col_names_col = column_names[col]
        if isinstance(col_names_col, list):
            column_names_list += col_names_col
        else:
            column_names_list.append(col_names_col)

    X = np.concatenate(X, axis = 1)

    assert len(column_names_list) == X.shape[1], "There are more columns than column names"

    return X, y, column_names_list, encoder_dict

    
