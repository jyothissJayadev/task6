# Install NLTK and download VADER lexicon if not already installed
import nltk
nltk.download('vader_lexicon')

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Sample time series data
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = np.random.rand(365) * 100  # Random values for demonstration
df = pd.DataFrame({'date': dates, 'value': values})
df.set_index('date', inplace=True)

# Perform seasonal decomposition
result = seasonal_decompose(df['value'], model='additive', period=30)  # Assuming weekly seasonality

# Plot the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df['value'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(result.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(result.seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(result.resid, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Sample dataset of customer reviews
data = {
    'review': [
        'Great product! Very happy with it.',
        'Average quality, but good price.',
        'Terrible service. Would not recommend.',
        'Amazing experience. Highly recommended!'
    ]
}

df_reviews = pd.DataFrame(data)

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment_score(review):
    scores = sid.polarity_scores(review)
    return scores['compound']

# Apply sentiment analysis to each review
df_reviews['sentiment_score'] = df_reviews['review'].apply(get_sentiment_score)

# Display the results
print("Customer Reviews with Sentiment Scores:")
print(df_reviews)
