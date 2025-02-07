import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class ModelPredictor:
    """
    Text classification predictor using a saved model
    model_path : Path to the saved model file
    """
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.vectorizer = self.model.named_steps['tfidf']

    def predict_df(self, df, text_column='text'):
        """
        Predict sentiment for a DataFrame
        df : DataFrame containing the text column to predict sentiment for
        text_column : str, default='text'
            Name of the text column in the DataFrame
            
        Returns DataFrame with original data and predicted sentiments
        """
        # Vectorize the input tweets using the same vectorizer used during training
        texts = df[text_column]
        texts_vect = self.vectorizer.transform(texts)
        
        # Predict the tweets sentiment
        predictions = self.model.predict(texts_vect)
        
        # Create a new DataFrame with the original data and predictions
        result_df = df.copy()
        result_df['sentiment'] = predictions
        
        return result_df

    def print_predictions(self, df, label_mapping, Sentiment_faces):
        """
        Parameters
        df : pd.DataFrame
            DataFrame containing the text column and predicted sentiments
        label_mapping : dict
            Mapping from numeric labels to string labels
        Sentiment_faces : dict
            Mapping from string labels to emojis
        """
        df['sentiment_str'] = df['sentiment'].map(label_mapping)
        for _, row in df.iterrows():
            tweet = row['text']
            sentiment = row['sentiment_str']
            emoji = Sentiment_faces[sentiment]
            print('Tweet:', tweet)
            print('Sentiment:', sentiment)
            print('Emoji:', emoji)
            print('----------------------------')