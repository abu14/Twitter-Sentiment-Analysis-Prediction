# **Twitter Sentiment Analysis & Prediction**

![GitHub contributors](https://img.shields.io/github/contributors/abu14/Twitter-Sentiment-Analysis-Prediction)
![GitHub forks](https://img.shields.io/github/forks/abu14/Twitter-Sentiment-Analysis-Prediction?style=social)
![GitHub stars](https://img.shields.io/github/stars/abu14/Twitter-Sentiment-Analysis-Prediction?style=social)
![GitHub issues](https://img.shields.io/github/issues/abu14/Twitter-Sentiment-Analysis-Prediction)
![GitHub license](https://img.shields.io/github/license/abu14/Twitter-Sentiment-Analysis-Prediction)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/abenezer-tesfaye-191579214/)


<!-- Table of Contents -->
## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tools Used](#tools-used)
- [Development Workflow](#development-workflow)
- [Licence](#license)
- [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## Project Overview
This project aims to analyze the sentiment of tweets using machine learning techniques. The primary goal is to classify tweets as positive, negative, or neutral based on their content. The process involves several key steps: data collection, preprocessing, feature extraction, model training, and evaluation. Tweets are collected using the Twitter API and stored in a DataFrame. Preprocessing includes text normalization, tokenization, removal of stopwords, and vectorization using TF-IDF. A machine learning model, such as a logistic regression or a random forest classifier, is trained on the preprocessed data. The model's performance is evaluated using metrics like accuracy, precision, recall, and F1-score. The final model is then used to predict the sentiment of new tweets, providing valuable insights into public opinion on various topics.

<!-- Architecture -->
## Architecture
The project consists of two main architectures, each containing specific pipelines for different purposes:

- #### Training Pipeline Architecture
For training the model and consists of these components:
1. Data Ingestion: Download the datasets from kaggle.
2. Preprocessing: Apply preprocessing techiniques to the datasets.
3. Data Exploration: Analyzing the various characterstics of the dataset.
4. Model Training: Builds and trains the ship detection model.
5. Model Evaluation: Evaluates the performance of the trained model.
6. Model Deployment: Using a simple app built on Flask to showcase the model in action. 
<p align="center">
  <img src="project_workflow.PNG" alt="Project Workflow">
</p>

<!-- Tools Uses -->

## Tools Used
<p>
<img src="https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/-FastAPI-009688?style=flat&logo=fastapi&logoColor=white">
<img src="https://img.shields.io/badge/-Flask-000000?style=flat&logo=flask&logoColor=white">
<img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/-Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white">
<img src="https://img.shields.io/badge/-Seaborn-3888E3?style=flat&logo=seaborn&logoColor=white">
</p>


<!-- Tools Uses -->

## Development Workflow




<!-- LICENSE -->
## License
This project is licensed under the MIT License. See [LICENSE](./LICENCE) file for more details.



<!-- CONTACT -->
## Contact

##### ⭐️Abenezer Tesfaye

Email - tesfayeabenezer64@gmail.com
 
Project Link: [Github Repo](https://github.com/abu14/Twitter-Sentiment-Analysis-Prediction)
