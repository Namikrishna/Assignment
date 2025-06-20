from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample Documents
docs = [
    "Data science is fun and exciting",
    "Machine learning is a branch of data science",
    "Data analysis leads to insights"
]

# 1. TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(docs)

# 2. Word Cloud (TF-IDF Scores)
tfidf_scores = X_tfidf.toarray().sum(axis=0)
words = tfidf_vectorizer.get_feature_names_out()
tfidf_dict = dict(zip(words, tfidf_scores))

wordcloud = WordCloud(background_color='white').generate_from_frequencies(tfidf_dict)
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("TF-IDF Word Cloud")
plt.show()

# 3. Cosine Similarity Matrix
similarity_matrix = cosine_similarity(X_tfidf)
df_sim = pd.DataFrame(similarity_matrix,
                      columns=[f'Doc {i+1}' for i in range(len(docs))],
                      index=[f'Doc {i+1}' for i in range(len(docs))])

# 4. Heatmap Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(df_sim, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Document Similarity Matrix (TF-IDF)")
plt.show()
