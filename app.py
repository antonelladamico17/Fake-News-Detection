import streamlit as st
import os
from urllib.request import urlopen
from bs4 import BeautifulSoup
from unidecode import unidecode
from nltk.tokenize import word_tokenize
import re
from urllib.parse import urlparse
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = nltk.corpus.stopwords.words('english')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pandas as pd

os.chdir(os.path.dirname(os.path.realpath(__file__)))

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





def main():
	st.title("Fake News Detector")
	st.write("__________________")

	user_input = st.text_input("Enter URL:")
	if user_input == '':
		st.text('Enter URL, plese')
	else:
		soup = BeautifulSoup(urlopen(user_input))
		
		df = [user_input]
		df = pd.DataFrame(df, columns = ['url'])
		df['title'] = soup.title.get_text()
		
		df['title'] = df['title'].apply(lambda x: clean_text(x))
		df["parsed_url"] = df.url.apply(parse_url)
		df = pd.concat([
			df.drop(['parsed_url'], axis=1),
			df['parsed_url'].apply(pd.Series)
		], axis=1])
		
				
				
		st.text(str(df['title']))
		st.text(str(df["parsed_url"]))
		st.text(str(df))

if __name__ == '__main__':
	main()

	
