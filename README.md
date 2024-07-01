
# Emotion Detection README
# Introduction
This project aims to build a system that detects emotions from text data using machine learning and deep learning techniques. The dataset contains text comments labeled with different emotions, and the project involves text preprocessing, feature extraction, model building, and evaluation using both traditional machine learning algorithms and a deep learning LSTM model.

# Before Proceeding
Before proceeding, ensure you have the necessary libraries installed. If they are already installed, you can ignore the following commands:

!pip install tensorflow==2.15.0
!pip install scikit-learn
!pip install pandas
!pip install numpy
!pip install seaborn
!pip install matplotlib
!pip install wordcloud
!pip install nltk
!pip install keras-preprocessing
Load Libraries
The project uses various libraries for deep learning, machine learning, and data visualization. These include Keras for deep learning models, scikit-learn for machine learning models, pandas for data manipulation, and seaborn and matplotlib for data visualization. WordCloud is used for creating word cloud visualizations, and NLTK is used for natural language processing tasks.

# Dataset
The dataset consists of text comments labeled with emotions. The data is loaded into a pandas DataFrame, and initial preprocessing steps include checking for missing values and duplicates, as well as computing the length of each comment.

# Data Preprocessing
Encode Emotions
The emotions in the dataset are encoded using LabelEncoder from scikit-learn to convert the categorical emotion labels into numerical format.

# Clean Text
Text data is cleaned by removing non-alphabetic characters, converting text to lowercase, splitting into words, and removing stopwords. Stemming is applied to reduce words to their root form.

# Train-Test Split
The cleaned text data is split into training and testing sets to evaluate the performance of the models.

# Vectorization
The text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features that can be fed into machine learning models.

# Model Building
Machine Learning Models
Several machine learning algorithms are applied to classify emotions, including:

Multinomial Naive Bayes
Logistic Regression
Random Forest Classifier
Support Vector Machine
Deep Learning Model
A Sequential model with LSTM layers is built using Keras. The model includes an embedding layer to convert input sequences to dense vectors, followed by LSTM and dense layers. The model is trained to predict emotions from text data.

# Model Evaluation
The performance of the models is evaluated using accuracy scores and classification reports. Various metrics such as precision, recall, and F1-score are used to assess the effectiveness of the models in classifying emotions.

# Inference
A predictive system is implemented to classify new text inputs into emotions. The text is preprocessed and vectorized before being fed into the trained model for prediction. The predicted emotion and its associated probability are outputted.

# Saving the Model
The trained machine learning and deep learning models, along with the label encoder and vectorizer, are saved using pickle. The deep learning model is also saved in HDF5 format for later use.

# Conclusion
This project demonstrates the process of building an emotion detection system using text data. It covers the entire pipeline from data preprocessing and feature extraction to model building, evaluation, and inference. The combination of machine learning and deep learning techniques provides robust and accurate emotion classification from text commen