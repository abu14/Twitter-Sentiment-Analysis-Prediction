# **Twitter Sentiment Analysis & Prediction**

![GitHub contributors](https://img.shields.io/github/contributors/abu14/Twitter-Sentiment-Analysis-Prediction)
![GitHub forks](https://img.shields.io/github/forks/abu14/Twitter-Sentiment-Analysis-Prediction?style=social)
![GitHub stars](https://img.shields.io/github/stars/abu14/Twitter-Sentiment-Analysis-Prediction?style=social)
![GitHub issues](https://img.shields.io/github/issues/abu14/Twitter-Sentiment-Analysis-Prediction)
![GitHub license](https://img.shields.io/github/license/abu14/Twitter-Sentiment-Analysis-Prediction)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/abenezer-tesfaye-191579214/)


<!-- Table of Contents -->
## **Table of Contents**
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tools Used](#tools-used)
- [Development Workflow](#development-workflow)
- [Licence](#license)
- [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## **Project Overview**
This project aims to analyze the sentiment of tweets using machine learning techniques. The primary goal is to classify tweets as positive, negative, neutral, or irrelevant based on their content. The process involves several key steps: data ingestion, preprocessing, model training, and evaluation. Tweets are collected using the provided Kaggle dataset with a training and validation dataframe. Preprocessing includes text normalization, tokenization, removal of stopwords, and vectorization using TF-IDF. A variety of machine learning algorithms, such as support vector machines, random forest classifier, and others were trained on the preprocessed data. The models performance were evaluated using metrics like accuracy, precision, recall, and F1-score. The final and best performing model is then used to predict the sentiment of new tweets, providing valuable insights into public opinion on various topics.

<!-- Architecture -->
## **Architecture**
The project consists of two main architectures, each containing specific pipelines for different purposes:

- #### Training Pipeline Architecture
For training the model and consists of these components:
1. **Data Ingestion**: Download the datasets from kaggle.
2. **Preprocessing**: Apply preprocessing techiniques to the datasets.
3. **Data Exploration**: Analyzing the various characterstics of the dataset.
4. **Model Training**: Builds and trains the ship detection model.
5. **Model Evaluation**: Evaluates the performance of the trained model.
6. **Model Deployment**: Using a simple app built on Flask to showcase the model in action. 
<p align="center">
  <img src="project_workflow.PNG" alt="Project Workflow">
</p>

<!-- Tools Uses -->

## **Tools Used**
<p>
<img src="https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/-FastAPI-009688?style=flat&logo=fastapi&logoColor=white">
<img src="https://img.shields.io/badge/-Flask-000000?style=flat&logo=flask&logoColor=white">
<img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/-Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white">
<img src="https://img.shields.io/badge/-Seaborn-3888E3?style=flat&logo=seaborn&logoColor=white">
<img src="https://img.shields.io/badge/-joblib-000000?style=flat&logo=joblib&logoColor=white">
<img src="https://img.shields.io/badge/-HTML-E34F26?style=flat&logo=html5&logoColor=white">
<img src="https://img.shields.io/badge/-NLTK-000000?style=flat&logo=nltk&logoColor=white">
</p>


<!-- Tools Uses -->

## **Development Workflow**

In this section we'll cover the most critical elements of the development process. We'll cover the four most imiportant sections in this project. But be sure to check out the source codes and notebook to get a more closer look.

### Data Preprocessing

The data preprocessing pipeline consists of several steps designed to clean and transform raw text data into a format suitable for machine learning models. These steps include:

- **Normalization:** Convert all text to lowercase.
- **Filtering:** Remove null values and convert the column to string type.
- **HTML Removal:** Eliminate any HTML tags from the text.
- **URL Removal:** Strip out URLs to focus on the main content.
- **Number Removal:** Remove numerical values that may not be relevant for sentiment analysis.
- **Punctuation Removal:** Eliminate punctuation marks to simplify the text.
- **Tokenization:** Break down the text into individual words (tokens).
- **Stopword Removal:** Filter out common words that don't carry much meaning.
- **Emoji Removal:** Remove emojis to focus on textual content.
- **Vectorization:** Convert the preprocessed text into numerical features using TF-IDF vectorization.

These steps ensure that the input data is clean and ready for model training.

#### Code Snippet
```python
class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def normalize_text(self, column):
        self.df[column] = self.df[column].apply(lambda text: text.lower())
        return self.df
# -------------------------

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
```

#### Model Training

The `Model Training` is designed to handle the full training process for a text classification model. It includes steps for data preparation, model training, and evaluation. This provides a streamlined way to train a Random Forest Classifier using TF-IDF vectorization and ensures reproducibility through configurable parameters.

Key features:

- **Label Encoding:** Convert target variables into numerical labels.
- **Data Splitting:** Divide the dataset into training and testing sets.
- **Pipeline Construction:** Build a machine learning pipeline that combines TF-IDF vectorization and a Random Forest Classifier.
- **Model Training:** Train the model on the training data.
- **Evaluation:** Generate a classification report to assess the model's performance.


#### Code Snippet
```python
    def train(self):
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
            ('clf', RandomForestClassifier(random_state=self.random_state))])
```

#### Deployment

This Flask application provides a web interface for performing sentiment analysis on text data. It allows users to upload a CSV file containing text data, preprocesses the data, makes predictions using a trained machine learning model, and displays the results in a user-friendly format.

Key features:

- **File Upload:** Users can upload a CSV file containing text data.
- **Data Preprocessing:** The uploaded data is preprocessed using the same steps as during training.
- **Prediction:** Predictions are made using a pre-trained Random Forest Classifier.
- **Result Display:** The prediction results are displayed in an HTML table for easy viewing.



#### Code Snippet
```python
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(file)
        # Predict sentiments
        result_df = predictor.predict_df(df, text_column='text')
        # Convert to HTML
        result_html = result_df.to_html(index=False)
        
        return render_template('result.html', result=result_html)

```

<!-- LICENSE -->
## **License**
This project is licensed under the MIT License. See [LICENSE](./LICENSE) file for more details.



<!-- CONTACT -->
## **Contact**

##### Abenezer Tesfaye

⭐️ Email - tesfayeabenezer64@gmail.com
 
Project Link: [Github Repo](https://github.com/abu14/Twitter-Sentiment-Analysis-Prediction)
