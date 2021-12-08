"""
Several modules to perform text preprocessing,
lower_casing --> remove_special_symbols --> remove_punctuation --> remove_stopwords
"""

import re
import spacy
import nltk
import string as st
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
# load small spacy model
nlp = spacy.load("en_core_web_sm")

# instantiate WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# convert string into its lower case
def lower_casing(text):
    """
    Obtained the lower case version and passed lower case version of text 

    Arguments:
    text: raw text, strings

    Returns:
    lower_text: string, representing lower case of raw text
    """

    # lower casing
    lower_text = text.lower()

    return lower_text

# remove special symbols using regex
def remove_special_symbols(text):
    """
    Removes special symbols and punctuation using regex
    """
#     text = re.sub("[●|•|,|\n|\\\\n|\/|(|)|-|–]", " ", text)
    text = re.sub("\n|\\\\n|●|•|○|,|\/|-|–|\(|\)||\\\\", " ", text)
    
    # replace multiple white spaces with single white space
    text = re.sub(" +", " ", text)
    
    # Removing punctuations in string
    # Using regex, replace all symbols except 0to9, atoZ, _, and white space
    # \w: returns a match where the string contains any word characters (character from a to Z, digits 0 to 9, _ character)
    # \s: returns a match where the string contains a white space character
    # ^ inside [] indicates negation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# remove punctuations, WOrk on code optimization (remove redundant for loop)
def remove_punctuation(text):
    """
    removes punctuation symbols present in the raw text passed as an arguments
    
    Arguments:
    text: raw text
    
    Returns: 
    not_punctuation: list of tokens without punctuation
    """

    return ("".join([ch for ch in text if ch not in st.punctuation]))
    # passing the text to nlp and initialize an object called 'doc'
    # doc = nlp(text)
    
    
    # not_punctuation = []
    # # remove the puctuation
    # for token in doc:
    #     if token.is_punct == False:
    #         not_punctuation.append(token)
    # return [item.text for item in not_punctuation if item.text not in []]


# remove stopwords
def remove_stopwords(tokens):
    """
    Removes stopwords passed from the tokens list passed as an arguments
    
    Arguments:
    tokens: list of tokens
    
    Returns:
    tokens_without_sw: list of tokens of raw text without stopwords
    """
    
    # getting list of default stop words in spaCy english model
    stopwords =nlp.Defaults.stop_words
    
    # tokenize text
    text_tokens = tokens
    
    # remove stop words:
    tokens_without_sw = [word for word in text_tokens if word not in stopwords]
    
    # return list of tokens with no stop words
    return tokens_without_sw

# Remove tokens of length less than 3
def remove_small_words(text):
    return [x for x in text if len(x) > 2]
# Stemming
# def stemming(tokens):
#     stem_list = []
#     for token in tokens:
#         stem_list.append(stemmer.stem(token))
#     return stem_list

# Lemmatization
def lemmatization(tokens):
    """
    obtain the lemma of the each token in the token list, append to the list, and returns the list
    
    Arguments:
    text: list of tokens
    
    Returns:
    lemma_list: return list of lemma corresponding to each tokens
    """
    

    lemma_list = []
    # Lemmatization
    for token in tokens:
        lemma_list.append(lemmatizer.lemmatize(token))
    
    return lemma_list


# preprocessing pipeline
# lower_casing --> remove_special_symbols --> remove_punctuation --> remove_stopwords
def preprocess_text(text):
    """
    - preprocess raw text passed as an arguments
    - preprocessing of text includes, lower_casing, remove_special_symbols, remove_punctuation, remove_stopwords
    
    Arguments: raw text, string
    
    Returns: list of tokens obtained after preprocessing
    """     
     # lower casing
    lower_case_text = lower_casing(text)

    # remove special symbols 
    removed_special_symbols = remove_special_symbols(lower_case_text)

    # remove punctuations
    # removed_punctuations = remove_punctuation(removed_special_symbols)

     # remove stopwords
    tokens_without_stopwords = remove_stopwords(removed_special_symbols.split(" "))

    tokens_with_small_words_removed = remove_small_words(tokens_without_stopwords)

    # lemmatization
    lemma_of_tokens = lemmatization(tokens_with_small_words_removed)
    return lemma_of_tokens
    # return tokens_without_stopwords
