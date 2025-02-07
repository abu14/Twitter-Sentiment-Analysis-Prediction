import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def normalize_text(self, column):
        """
        Purpose:
            lowercase (normalise) values in the dataframe
        Retruns:
            lowercased tweets dataframe
        """
        self.df[column] = self.df[column].apply(lambda text: text.lower())
        return self.df
        
    def filter_strings(self, column):
        """
        Purpose:
            Drops null values and also convert datatype to str column provided in the dataframe
        Retruns:
            Cleaned dataframe
        """
        self.df = self.df.dropna(subset=[column])
        self.df[column] = self.df[column].astype(str)
        return self.df

    def remove_html(self, column):
        """
        Purpose:
            remove HTML values in the dataframe
        Retruns:
            cleaned column provided
        """
        self.df[column] = self.df[column].apply(lambda text: re.sub(r'<.*?>','',text))
        return self.df

    def remove_urls(self, column):
        """
        Purpose:
            Remove URLs in the column provided
        Retruns:
            Cleaned column/ DataFrame
        """
        self.df[column] = self.df[column].apply(lambda text: re.sub(r'http\S+|www\S+', '', text))
        return self.df

    def remove_num_values(self, column):
        """
        Purpose:
            Remove URLs in the column provided
        Retruns:
            Cleaned column/ DataFrame
        """
        self.df[column] = self.df[column].apply(lambda text: re.sub(r'\d+', '', text))
        return self.df

    def remove_punct(self, column):
        """
        Purpose:
            Remove punctuation in the column provided
        Retruns:
            Cleaned column/ DataFrame
        """
        self.df[column] = self.df[column].apply(lambda text: text.translate(str.maketrans('','', string.punctuation)))
        return self.df

    def tokenizer_text(self, column):
        """
        Purpose:
            Break down the strings to tokens in the column provided
        Retruns:
            Cleaned column/ DataFrame
        """
        self.df[column] = self.df[column].apply(lambda text: word_tokenize(text))
        return self.df

    def remove_stopwords(self, column):
        """
        Purpose:
            Remove value that frequently occur and have no impact from the column provided
        Retruns:
            Cleaned column/ DataFrame
        """
        stop_words = set(stopwords.words('english'))
        self.df[column] = self.df[column].apply(lambda text: ' '.join
                                                ([word for word in word_tokenize(text.lower()) if word not in stop_words]))
        return self.df
    
    def remove_emojis(self, column):
        """
        Purpose:
            Remove Emojis from the column provided
        Retruns:
            Cleaned column/ DataFrame
        """
        emoji = re.compile(
                "["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002500-\U00002BEF"  # chinese char
                    u"\U00002702-\U000027B0"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    u"\U0001f926-\U0001f937"
                    u"\U00010000-\U0010ffff"
                    u"\u2640-\u2642"
                    u"\u2600-\u2B55"
                    u"\u200d"
                    u"\u23cf"
                    u"\u23e9"
                    u"\u231a"
                    u"\ufe0f"  # dingbats
                    u"\u3030"
                    "]+",
                    flags=re.UNICODE
                )
        self.df[column] = self.df[column].apply(lambda text: emoji.sub(r'', text))
        return self.df    

    def vectorize_df(self, text_data):
        """
        Purpose:
            Vectorize the column provided
        Retruns:
            Cleaned column/ DataFrame
        """
        text_strings = [" ".join(tokens) for tokens in text_data]
        tfidvectorizer = TfidfVectorizer()
        tfidvectors = tfidvectorizer.fit_transform(text_strings)
        return tfidvectors, tfidvectorizer

    def join_tweets(self, column):
        self.df[column] = self.df[column].apply(lambda x: ' '.join(map(str, x)))
        return self.df

    def preprocesser(self, column):
        """Applies preprocessing steps to the DataFrame."""
        self.normalize_text(column)
        self.filter_strings(column)
        self.remove_html(column)
        self.remove_urls(column)
        self.remove_num_values(column)
        self.remove_stopwords(column)
        self.remove_punct(column)
        self.tokenizer_text(column)
        self.join_tweets(column)
        return self.df

        """
        How to Preprocess, sample:
        
        - Initialize the object
        
        preprocessor_ = DataPreprocessor(df)
        
        - Preprocess the data using the new method
        
        preprocessor_df = preprocessor_.preprocesser('Tweet')

         - Finally you should see merged the tweets using join_tweets
          eg. coming borders kill
        """