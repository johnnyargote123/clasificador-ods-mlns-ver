import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords', quiet=True)

spanish_stopwords_set = set(stopwords.words('spanish'))
spanish_stemmer = SnowballStemmer('spanish')
tokenizer = RegexpTokenizer(r'\w+')

def preprocess_text(text):
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in spanish_stopwords_set]
    tokens = [spanish_stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)