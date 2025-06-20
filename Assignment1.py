from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create BoW model

docs = [
    "Data science is fun and exciting",
    "Machine learning is a branch of data science",
    "Data analysis leads to insights"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
bow_freq = X.toarray().sum(axis=0)
words = vectorizer.get_feature_names_out()
bow_dict = dict(zip(words, bow_freq))

# Plot Word Cloud
wordcloud = WordCloud(background_color='white').generate_from_frequencies(bow_dict)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Bag-of-Words Word Cloud")
plt.show()
