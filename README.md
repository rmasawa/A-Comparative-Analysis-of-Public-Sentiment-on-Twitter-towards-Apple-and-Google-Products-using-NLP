# Sentiment Analysis of Tweets about Apple and Google Products
![Sentiment Analysis](https://github.com/rmasawa/A-Comparative-Analysis-of-Public-Sentiment-on-Twitter-towards-Apple-and-Google-Products-using-NLP/blob/master/Images/Sentiment_Analysis_Projects.png)

##  Authors

1. [Rose Miriti]()
2. [Isaac Wadhare]()
3. [Lydia Chumba]()
4. [Marilyn Akinyi]()
5. [Rodgers Otieno]()
6. [Erick Mauti]()
7. [Samwel Ongechi]()

##  Project Overview

- The aim of this project is to build a Natural Language Processing (NLP) model that can classify tweets about Apple and Google products into Positive, Negative, or Neutral sentiments.
- By leveraging machine learning techniques, the project provides a scalable way to analyze public opinion from social media, offering valuable insights into customer perceptions and brand sentiment.
- The ultimate goal is to support decision-making and market understanding for technology companies.

## Project Objectives

1. Determine the overall public sentiment towards Apple and Google products on Twitter.
2. Identify tweet characteristics and themes that contribute to positive, negative, or neutral sentiment.
3. Provide actionable insights from sentiment trends to inform business decisions, marketing strategies, and product improvements.

## Project Workflow

### Data Source
The dataset comes from [CrowdFlower](https://data.world/crowdflower/brands-and-product-emotions) and contains over 9,000 tweets labeled as positive, negative, or neutral.

### Exploratory Data Analysis (EDA)
- Inspected structure, missing values, and duplicates
- Checked class distribution (Neutral: 61%, Positive: 33%, Negative: 6%)
- Compared Apple vs. Google mentions (Apple dominates ~85% more)
- Created WordClouds and histograms

![Sentiment Distribution](https://github.com/rmasawa/A-Comparative-Analysis-of-Public-Sentiment-on-Twitter-towards-Apple-and-Google-Products-using-NLP/blob/master/Images/Sentiment%20Distribution.png)
![Emotions towards brands](https://github.com/rmasawa/A-Comparative-Analysis-of-Public-Sentiment-on-Twitter-towards-Apple-and-Google-Products-using-NLP/blob/master/Images/Emotions%20towards%20brands.png)

### Data Preprocessing
- Cleaning: Removed URLs, mentions, hashtags, numbers, special characters
- Tokenization & Stopword Removal
- Lemmatization
- Vectorization: TF-IDF + N-grams
- Label Encoding

![Text Length Analysis](https://github.com/rmasawa/A-Comparative-Analysis-of-Public-Sentiment-on-Twitter-towards-Apple-and-Google-Products-using-NLP/blob/master/Images/Text%20length%20analysis.png)

### Feature Engineering
- TF-IDF
- N-grams (bi-grams & tri-grams)

### Modeling
Tested classifiers:
- Logistic Regression
- Random Forest
- Neural Networks
- XGBoost (chosen)

Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-score
- Classification reports

### Why XGBoost Was Chosen
- Best Accuracy & Generalization – 66.9% test accuracy, lowest overfitting gap (13.8%)
- Handles Class Imbalance – Negative tweets were underrepresented
- Efficiency – Optimized, scalable, fast training
- Interpretability – Provides feature importance
- Industry-Proven – Widely adopted in NLP & Kaggle

Conclusion: XGBoost was selected due to its balanced performance and robustness.

[View XGBoost Model Results](https://github.com/rmasawa/A-Comparative-Analysis-of-Public-Sentiment-on-Twitter-towards-Apple-and-Google-Products-using-NLP/blob/master/Index.ipynb)


## Recommendations  

1. **Deploy XGBoost Model** – Achieved best generalization with competitive accuracy and minimal overfitting.  
2. **Monitor Sentiment Trends Over Time** – Track daily/weekly results to detect spikes in negative sentiment.  
3. **Implement Periodic Retraining** – Continuously update the model with new tweets to capture evolving slang, emojis, and trends.  
4. **Expand Analysis** – Future work could explore deep learning models like **Word2Vec** or **BERT** for better context handling.  

## Limitations  

1. **Class Imbalance** – Negative tweets are underrepresented, leading to lower recall for that class.  
2. **Short Texts** – Tweets average 6–11 words, limiting context for sentiment detection.  
3. **Dynamic Language** – Social media slang and emoji usage evolve rapidly, reducing long-term generalization.  
4. **Context Shifts** – Trained on historical data, the model may struggle with sudden changes in sentiment (e.g., product launches).  
5. **Performance Ceiling** – Traditional ML models plateau around ~68% accuracy; future iterations should test transformer-based models.  

## Visualizations  

- [Tableau Dashboard 1](https://public.tableau.com/views/SentimentAnalysis_17568185002180/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)  
- [Tableau Dashboard 2](https://public.tableau.com/app/profile/rose.miriti/viz/Phase_4_17567508830150/Dashboard1)  

### Deployed Model
The trained XGBoost model is deployed using joblib and a Flask app (`deploy_sentiment_app.py`). Users can input a tweet and get the predicted sentiment (Positive / Negative / Neutral).

#### Run Locally
```bash
git clone https://github.com/rmasawa/A-Comparative-Analysis-of-Public-Sentiment-on-Twitter-towards-Apple-and-Google-Products-using-NLP.git
cd A-Comparative-Analysis-of-Public-Sentiment-on-Twitter-towards-Apple-and-Google-Products-using-NLP
pip install -r requirements.txt
python deploy_sentiment_app.py

