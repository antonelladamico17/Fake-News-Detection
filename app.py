import streamlit as st
import os
import pandas as pd
import tldextract
import text2emotion as te
from urllib.request import urlopen
from bs4 import BeautifulSoup
from unidecode import unidecode
from nltk.tokenize import word_tokenize, RegexpTokenizer
import re
from urllib.parse import urlparse
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
smote = SMOTE()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()


os.chdir(os.path.dirname(os.path.realpath(__file__)))

DATA_PATH = 'pickle'

st.set_page_config(page_title="Fake News detector", layout="centered")

def clean_text(text):

  text = text.lower()
  text = text.strip() # Remove spaces at the beginning and at the end of the string
  text = re.sub("([0-9]+(?:st|nd|rd|th)+)", " ", text) # Remove ordinal numbers
  text = re.sub("(?!5g)[^a-zA-Z]", " ", text) # Remove numbers and punctuation. Exclude 5g because it is part of our main search keyword
  text = re.sub("\s+", " ", text) # Remove extra space
  text = re.sub("\d+(\.\d+)?","num", text)
  text = re.sub("s/^\s+.*\s+$//", " ", text) # Remove leading and trailing whitespaces
  text = re.sub(r"\b[a-zA-Z]\b", " ", text) # Remove single characters

  text = unidecode(re.sub("\s+", " ", text.strip())) # Remove any additional whitespace
  text = text.strip()
  
  text = text.replace("numg", str('fiveg')) # Replace the trasformation of 5G -> numg -> with fiveg in order to appear in the df

  tokenized_texts = []

  document = word_tokenize(text)

  for token in document:
    if token not in stop_words and len(token) > 1:

      tokenized_texts.append(lemmatizer.lemmatize(token))
  
  tokenized_texts = ' '.join([w for w in tokenized_texts if len(w) > 2 ])

  return tokenized_texts


def parse_url(url: str):
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



def emotion_detection(sents):
    """Main algo for convertion for the 5 emotions """
    sent_emotion = te.get_emotion(sents) # prende il testo 
    return sent_emotion


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


def get_data(path):
	domain_fake = pickle.load(open(Path(path, 'domain_fake.pkl'), "rb"))
	domain_real = pickle.load(open(Path(path, "domain_real.pkl"), "rb"))
	model = pickle.load(open(Path(path, "svc_clf.pkl"), "rb"))
				  
	return domain_fake, domain_real, model

class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

def main():
	st.title("Fake News Detector")
	st.write("__________________")
	
	st.sidebar.write('Credits:')
	st.sidebar.write("""
		Univesity of Milan\\
		M.Sc. in Data Science and Economics\\
		Thesis Title:\\
		Fighting against infodemic: \\
		Fake News detection in the Era of Covid-19\\
		Antonella D'Amico - 961150\\
		Academic Year: 2020/2021
		""")
	

	user_input = st.text_input("Enter URL:")
	submit = st.button('Check')
	
			
	if user_input!="" and (user_input.startswith('http://') or user_input.startswith('https://')):
		try:
			soup = BeautifulSoup(urlopen(user_input))


			df = [user_input]
			df = pd.DataFrame(df, columns = ['url'])
			df['title'] = soup.title.get_text()

			df['title'] = df['title'].apply(lambda x: clean_text(x))
			df["parsed_url"] = df.url.apply(parse_url)
			df = pd.concat([
				df.drop(['parsed_url'], axis=1),
				df['parsed_url'].apply(pd.Series)
			], axis=1)

			emotion_list = []
			for i, row in df.iterrows():
				emotion_dict = emotion_detection(row[1])
				emotion_list.append(emotion_dict)
			emotion_df = pd.DataFrame(emotion_list)
			df = pd.concat([df, emotion_df], axis = 1)
			polarity_Score = []
			for i, row in df.iterrows():
				score = sid.polarity_scores(row[1])
				polarity_Score.append(score)
			polarity_Score = pd.DataFrame(polarity_Score)
			df = pd.concat([df, polarity_Score], axis = 1)
			df = df.drop(['compound'], axis = 1)


			df["length"] = df.url.str.len()
			df["tld"] = df.netloc.apply(lambda nl: tldextract.extract(nl).suffix)
			df['tld'] = df['tld'].replace('','None')
			df['slashes'] = df.path.str.count('/')
			df['digit'] = df.url.str.count('\d')
			df['hypen'] = df.url.str.count('-')

			df['num_subdomains'] = df['netloc'].apply(lambda net: get_num_subdomains(net))
			df['domain_tokens'] = df['netloc'].apply(lambda net: tokenize_domain(net))
			df['path_tokens'] = df['path'].apply(lambda path: " ".join(map(str,tokenizer.tokenize(path))))


			df.drop('url', axis=1, inplace=True)
			df.drop('scheme', axis=1, inplace=True)
			df.drop('netloc', axis=1, inplace=True)
			df.drop('path', axis=1, inplace=True)
			df.drop('params', axis=1, inplace=True)
			df.drop('query', axis=1, inplace=True)
			df.drop('fragment', axis=1, inplace=True)
			df.drop('title', axis = 1, inplace = True)

			numeric_features = ['length', 'slashes', 'digit', 'hypen', 'Happy', 'Angry', 'Surprise', 'Sad', 'Fear', 'neg', 'neu', 'pos']
			numeric_transformer = Pipeline(steps=[
			    ('scaler', MinMaxScaler())])

			categorical_features = ['tld']
			categorical_transformer = Pipeline(steps=[
			    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

			vectorizer_features = ['domain_tokens','path_tokens', 'num_subdomains']
			vectorizer_transformer = Pipeline(steps=[
			    ('con', Converter()),
			    ('tf', TfidfVectorizer())])

			preprocessor = ColumnTransformer(
			    transformers=[
				('num', numeric_transformer, numeric_features),
				('cat', categorical_transformer, categorical_features),
				('domvec', vectorizer_transformer, ['domain_tokens']),
				('pathvec', vectorizer_transformer, ['path_tokens'])
			    ])


			domain_fake, domain_real, model = get_data(DATA_PATH)



			for i in range(len(df)):
				if df['domain_tokens'][i] in list(domain_fake):
					st.write('The URL domain appears to be registered as a domain that publishes fake news.\\Please be careful when reading this news, it may contain some false information.')
				elif df['domain_tokens'][i] in list(domain_real):
					st.write('Great! This domain is registered as a domain that publishes reliable news.\\Anyway, always be careful when reading the news online.')

				else:
					pred = model.predict(df)
					if pred == 1:
						st.write('This news may contain false information, please read it carefully')
						st.write("Chek if this news is present here: [https://www.politifact.com/fake-news/]")
					else:
						st.write('This news is reliable')

			st.write("Read more about 'How to spot Fake news' : [https://www.factcheck.org/2016/11/how-to-spot-fake-news/]")
		
		except:
			st.text('This URL [user_input] is not valid.')
	
	else:
		st.write('Enter valid URL, please.')
				
		

if __name__ == '__main__':
	main()

	
