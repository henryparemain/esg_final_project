import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk.corpus
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import joblib
import string
from sklearn.model_selection import train_test_split
import re
import requests
import pdftotext
import io

# TEXT COLLECTION
#--------------------------

def html_parse(url):
    '''
    Description:
    Takes a url as an input and returns the html of a webpage, which can be used to extract specific tags.

    Inputs:
    url (string): url of the webpage

    Returns:
    soup (bs4 object): parsed html of the webpage
    '''

    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    return soup

def get_exchange_companylist(baseurl):
    '''
    Description:
    Takes in the url for the respective exchange landing page and returns a list of the names of all the companies available on said landing page.

    Inputs:
    baseurl (string): url of the exchange landing page

    Returns:
    company_names (list of strings): list of the names of all the companies available on the landing page
    '''

    soup_nyse = html_parse(baseurl)
    company_names = []
    a_href=soup_nyse.find_all('span', class_ = 'companyName')
    for tag in a_href:
        company_names.append(tag.get_text())
    return(company_names)

def get_exchange_company_list_url_compatible(company_names):
    '''
    Description:
    Inputs a list of company names and returns the list with any blank space replaced with a '+' to ensure the names can be used within a url.

    Inputs:
    company_names (list of strings): list of company names

    Returns:
    company_names_url_compatible (list of strings): list of company names with any blank space replaced with a '+'
    '''

    company_names_url_compatible = []
    for i in range(len(company_names)):
        company_names_url_compatible.append(company_names[i].replace(' ', '+'))
    return company_names_url_compatible


#  PREPROCESSING FUNCTIONS
#--------------------------

def numbers_to_words(text):
    '''
    Description:
    Takes a string as an input and converts all the numerical representation of numbers to string representations.

    Inputs:
    text (string): input text containing numerical representations of numbers

    Returns:
    string: the text with numerical representations of numbers replaced by string representations
    '''

    # Split the text into words
    words = text.split()

    # Find all numbers in the string using regex
    numbers = re.findall(r'\d+', text)

    # Convert each number to words
    number_words = []
    for num in numbers:
        try:
            num_word = num2words(int(num))
            number_words.append(num_word)
        except:
            continue

    # Replace the numbers with their word equivalents
    for i, num in enumerate(numbers):
        try:
            idx = words.index(num)
            words[idx] = number_words[i]
        except:
            continue

    return ' '.join(words)

def remove_camel(document):
    '''
    Description:
    Takes a string as an input and identifies camel case, splitting up words where camel case is identified.

    Inputs:
    document (string): input string containing camel case

    Returns:
    string: the input string with camel case split up into separate words
    '''

    li = []
    pattern = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])') #identifies camel case, but ingores pure upper case words.
    for word in document.split():
        # Split the string at camel case
        word_fix = pattern.split(word)
        # append
        li.append(' '.join(word_fix))
    return ' '.join(li)


def split_number_words(document):
    '''
    Takes in a string and identifies words which have numbers within the word, or appended to the front or back 
    of a word. Splits the string at these points.
    
    Inputs:
    document (string): input string containing words with numbers
    
    Returns:
    string: the input string with words containing numbers split up into separate words
    '''
    li = []
    pattern = re.compile(r'(\d+|[a-zA-Z]+)') 
    # Split the string at word or number boundaries
    document = pattern.findall(document)
    for word in document:
        if word.isdigit():
            li.append(word)
        else:
            # split words at number boundaries
            pattern = re.compile(r'(\d+|\D+)')
            word_parts = pattern.findall(word)
            for part in word_parts:
                if part.isdigit():
                    li.append(part)
                elif part.isalpha():
                    li.append(part)
    return ' '.join(li)



def remove_numbers(document):
    '''
    Removes all numbers from a string

    Inputs:
    document (string): input string with numbers to be removed

    Returns:
    string: the input string with all numbers removed
    '''
    extract = r'[0-9]'
    document= re.sub(extract, "", document) 
    return document


def remove_whitespace_tokens(document):
    '''
    Removes whitespace/empty tokens from document (string)

    Inputs:
    document (string): input string with whitespace/empty tokens to be removed

    Returns:
    list: a list of non-empty tokens in the input string
    '''
    document = [word for word in document if word!= '']
    return (document)


def remove_punctuation(document, punc):
    '''
    Removes punctuation provided (string) from document (string)

    Inputs:
    document (string): input string with punctuation to be removed
    punc (string): a string containing the punctuation to be removed

    Returns:
    string: the input string with the specified punctuation removed
    '''
    document1 = re.sub(f"\{punc}", '', document)
    
    return document1


def remove_all_punctuation(document):
    '''
    Removes all punctuation (found in string.punctuation) from document (string)
    
    Inputs:
    document (string): The input string to remove all punctuation from
    
    Returns:
    string: The input string with all punctuation removed
    '''
    for punc in string.punctuation:
        document1 = remove_punctuation(document, punc)
    
    return document1


def remove_stopwords(list_of_tokens):
    """
    Removes stopwords from a list of tokens
    
    Inputs:
    list_of_tokens (list): A list of tokens
    
    Returns:
    list: A list of tokens with stopwords removed
    """
    
    stop_words = stopwords.words('english')
    
    cleaned_tokens = [] 
    
    for token in list_of_tokens: 
        if token not in stop_words: 
            cleaned_tokens.append(token)
            
    return cleaned_tokens


def stemmer(list_of_tokens):
    '''
    Stems a list of tokens using the Porter Stemmer
    
    Inputs:
    list_of_tokens (list): A list of tokens
    
    Returns:
    list: A list of tokens that have been stemmed using the Porter Stemmer
    '''
    
    stemmed_tokens_list = []
    
    for i in list_of_tokens:
        token = PorterStemmer().stem(i)
        stemmed_tokens_list.append(token)
        
    return stemmed_tokens_list


def lemmatizer(list_of_tokens):
    '''
    Lemmatizes a list of tokens using the WordNetLemmatizer
    
    Inputs:
    list_of_tokens (list): A list of tokens
    
    Returns:
    list: A list of tokens that have been lemmatized using the WordNetLemmatizer
    '''
    
    lemmatized_tokens_list = []
    
    for i in list_of_tokens: 
        token = WordNetLemmatizer().lemmatize(i)
        lemmatized_tokens_list.append(token)
        
    return lemmatized_tokens_list


def noun_only(x):
    '''
    Filters a list of tokens to only include nouns using part of speech tagging
    
    Inputs:
    x (list): A list of tokens
    
    Returns:
    list: A list of tokens that are nouns according to part of speech tagging
    '''
    pos_comment = nltk.pos_tag(x)
    filtered = [word[0] for word in pos_comment if word[1] in ['NN','NNS','NNP','NNPS']]
    # to filter both noun and verbs
    #filtered = [word[0] for word in pos_comment if word[1] in ['NN','VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
    return filtered


def my_cleaner(document, min_len_tokens, noisy_words = [], lemmatization=False, stemming=False, nouns = False, camel = True, removeNums = True, lower = True,\
               newLineRemove=True, punc = True, uni = True , whitespace = True, stopwords =True):
    '''
    Function for use in cleaning text data. All cleaning functions are optional via Boolean toggling.

    Inputs:
    document (str): input string
    min_len_tokens (int): minimum length of tokens to keep
    noisy_words (list of str): words to exclude from the token list
    lemmatization (bool): apply lemmatization if True
    stemming (bool): apply stemming if True
    nouns (bool): only retain nouns if True
    camel (bool): remove camel case if True
    removeNums (bool): remove numbers if True
    lower (bool): lowercase all letters if True
    newLineRemove (bool): replace newline characters with space if True
    punc (bool): remove all punctuation if True
    uni (bool): remove unicode characters if True
    whitespace (bool): remove whitespace tokens if True
    stopwords (bool): remove stop words if True
    
    Returns:
    tokenized_document2 (list of str): list of cleaned tokens
    
    '''
    
    #Replace \n with ' '
    if newLineRemove:
        document = document.replace('\n', ' ')
    # remove camel case
    if camel:
        document= remove_camel(document)
    #remove numbers
    if removeNums:  
        document = remove_numbers(document)
    # lowercase
    if lower:
        document = str.lower(document)
    # remove punctuation
    if punc:
        document = remove_all_punctuation(document)
    # remove unicode
    if uni:
        document = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", document)
    # remove whitespace
    if whitespace:
        document = remove_whitespace_tokens(document.split())
    # tokenise the doc
    tokenized_document = document
    # remove stopwords
    if stopwords:
        tokenized_document = remove_stopwords(tokenized_document)
    #retain only nouns
    if nouns:
        tokenized_document = noun_only(tokenized_document)
        tokenized_document = remove_whitespace_tokens(tokenized_document)
    # lemmatization
    if lemmatization:
        tokenized_document = lemmatizer(tokenized_document)
        tokenized_document = remove_whitespace_tokens(tokenized_document)
    # stemming
    if stemming:
        tokenized_document = stemmer(tokenized_document)
        tokenized_document = remove_whitespace_tokens(tokenized_document)

    # iterate through the list of tokens and only return those greater than min_len_tokens
    tokenized_document2 = []
    for word in tokenized_document:
        if len(word) > min_len_tokens and word not in noisy_words:
            tokenized_document2.append(word)
        else:
            continue

    return (tokenized_document2)




def pdf_to_raw_text(url):
    """
    Converts a PDF file from a given URL to raw text
    
    Args:
    url (string): URL of the PDF file
    
    Returns:
    string: the raw text extracted from the PDF file
    """
    # Try to access url
    res = requests.get(url)
    # Use io.BytesIO to trick the script into thinking the pdf is in memory
    pdf = pdftotext.PDF(io.BytesIO(res.content))

    pdftotext_text = "\n\n".join(pdf)

    return pdftotext_text







#  LDA MODELLING FUNCTIONS
#--------------------------

import gensim 
import pyLDAvis
from gensim import corpora
pyLDAvis.enable_notebook()

def get_max_coherence(doc_term_matrix, dictionary, paras, range_of_topic_num):
    '''
    Calculate the coherence score for different number of topics and returns the number of topics
    that yield the highest coherence score.

    Parameters:
    -----------
    doc_term_matrix: scipy.sparse matrix
        The matrix containing the frequency of each term in each paragraph.
    dictionary: gensim.corpora.dictionary.Dictionary
        The dictionary used to get the numerical representation of each word.
    paras: pandas.Series
        The column of the dataframe containing cleaned and tokenized words within paragraphs.
    range_of_topic_num: list
        A list of integers representing the number of topics to be tested for coherence score.

    Returns:
    --------
    topic_number_with_max_coherence: int
        The number of topics that yield the highest coherence score.
    max_coherence: float
        The coherence score for the number of topics that yield the highest coherence score.
    '''
    coherence = []
    for k in range_of_topic_num:
        print('Round: '+str(k))
        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(doc_term_matrix, num_topics=k, id2word = dictionary, passes=40,\
                    iterations=200, chunksize = 10000, eval_every = None)
        
        cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=paras,\
                                                        dictionary=dictionary, coherence='c_v')
        coherence.append((k,cm.get_coherence()))
    
    # convert tuples to lists containing coherence score and topic numbers
    topic_number = [x[0] for x in coherence]
    coherence_score = [x[1] for x in coherence]
    #get index of max coherence score
    max_coherence_index = np.argmax(coherence_score)
    #get equivalent topic number
    topic_number_with_max_coherence = topic_number[max_coherence_index]
    max_coherence = coherence_score[max_coherence_index]
    
    return topic_number_with_max_coherence, max_coherence



# PLOTTING
#--------------------------
    
def plot_most_frequent(words, word_counts, top=20, title=None):
    """
    Plots a bar graph of the most frequent words in a given list of words and their respective counts.
    
    Args:
    words (list): list of words to plot
    word_counts (list): list of counts for each word
    top (int): number of most frequent words to plot (default: 20)
    title (string): title for the plot (default: None)
    
    Returns:
    None
    """
    words_df = pd.DataFrame({"token": words, 
                             "count": word_counts})
    
    fig, ax = plt.subplots(figsize=(0.75*top, 5))
    words_df.sort_values(by="count", ascending=False).head(top)\
        .set_index("token")\
        .plot(kind="bar", rot=45, ax=ax, color='g')
    sns.despine()
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()


    