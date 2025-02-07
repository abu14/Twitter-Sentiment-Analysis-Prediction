from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

class ModelTrainer:
    """
    Text classification trainer with persistence
    
    Parameters
    ----------
    feature_col : str
        Name of text feature column
    target_col : str
        Name of target variable
    test_size : float, default=0.2
        Proportion for test split
    random_state : int, default=42
        Reproducibility seed
    """
    def __init__(self, df,feature_col,target_col, test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.feature_col = feature_col
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline = None
        self.report = None

    def train(self):
        """
        Execute full training
        
        Parameters
        df : pd.DataFrame
            Input data containing features and target
        """
        # label Encoding the variables
        label_encoder = LabelEncoder()
        self.df[self.target_col] = label_encoder.fit_transform(self.df[self.target_col])
        
        #Feature and Target variables
        X = self.df[self.feature_col]
        y = self.df[self.target_col]

        
        #Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state
                )
        
        # Build pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(random_state=self.random_state))
                ])
        
        #vectorize and split the data
        self.pipeline.fit(X_train, y_train)

        #predict
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        self.report = report
     