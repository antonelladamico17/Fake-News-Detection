import pickle
import pandas as pd
import numpy as np
import random
import sys
import os
import tldextract
import warnings
import regex as re
from typing import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns 
from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer
from imblearn.over_sampling import SMOTE
smote = SMOTE()

warnings.filterwarnings("ignore")

DATA_PATH = 'pickle'

def parse_url(url: str) -> Optional[Dict[str, str]]:
    try:
        no_scheme = not url.startswith('https://') and not url.startswith('http://')
        if no_scheme:
            parsed_url = urlparse(f"http://{url}")
            return {
                "scheme": None, # not established a value for this
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
        else:
            parsed_url = urlparse(url)
            return {
                "scheme": parsed_url.scheme,
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
    except:
        return None
      
def get_num_subdomains(netloc: str) -> int:
    subdomain = tldextract.extract(netloc).subdomain 
    if subdomain == "":
        return 0
    return subdomain.count('.') + 1
  
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
def tokenize_domain(netloc: str) -> str:
    split_domain = tldextract.extract(netloc)
    no_tld = str(split_domain.subdomain +'.'+ split_domain.domain)
    return " ".join(map(str,tokenizer.tokenize(no_tld)))
  

if __name__ == '__main__':
        st.title("URL check")
    
        user_input = st.text_input("Enter URL:")
        prova = pd.DataFrame(user_input, columns = ['url'])
        
        prova["parsed_url"] = prova.url.apply(parse_url)
        prova = pd.concat([
            prova.drop(['parsed_url'], axis=1),
            prova['parsed_url'].apply(pd.Series)
        ], axis=1)
        
        prova["length"] = prova.url.str.len()
        prova["tld"] = prova.netloc.apply(lambda nl: tldextract.extract(nl).suffix)
        prova['tld'] = prova['tld'].replace('','None')
        prova['slashes'] = prova.path.str.count('/')
        prova['digit'] = prova.url.str.count('\d')
        prova['hypen'] = prova.url.str.count('-')
        
        prova['num_subdomains'] = prova['netloc'].apply(lambda net: get_num_subdomains(net))
        prova['domain_tokens'] = prova['netloc'].apply(lambda net: tokenize_domain(net))
        prova['path_tokens'] = prova['path'].apply(lambda path: " ".join(map(str,tokenizer.tokenize(path))))
        
        prova.drop('url', axis=1, inplace=True)
        prova.drop('scheme', axis=1, inplace=True)
        prova.drop('netloc', axis=1, inplace=True)
        prova.drop('path', axis=1, inplace=True)
        prova.drop('params', axis=1, inplace=True)
        prova.drop('query', axis=1, inplace=True)
        prova.drop('fragment', axis=1, inplace=True)
        
        model = pickle.load(open(Path(path, "svc_clf.pkl"), "rb"))
        
        
        if submit and user_input!="":
		    pred = model.predict(prova)
		    st.header("Type of URL : "+pred)
		    st.subheader("What is a "+pred+" URL?")
