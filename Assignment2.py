from sklearn.feature_extraction.text import CountVectorizer
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

# 1. Count Vectorization (BoW)
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(docs)

# 2. Print BoW Matrix
print("\n📦 Bag-of-Words Matrix:")
bow_df = pd.DataFrame(X_bow.toarray(), columns=bow_vectorizer.get_feature_names_out())
print(bow_df)

# 3. Word Cloud for Each Document
words = bow_vectorizer.get_feature_names_out()
fig, axs = plt.subplots(1, len(docs), figsize=(18, 4))

for i in range(len(docs)):
    bow_scores = X_bow[i].toarray().flatten()
    bow_dict = dict(zip(words, bow_scores))
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(bow_dict)
    
    axs[i].imshow(wordcloud, interpolation='bilinear')
    axs[i].axis('off')
    axs[i].set_title(f'Doc {i+1}')

plt.suptitle("BoW Word Clouds for Each Document")
plt.tight_layout()
plt.show()

# 4. Cosine Similarity Matrix
similarity_matrix = cosine_similarity(X_bow)

# 5. Print Similarity Matrix
print("\n🔗 Cosine Similarity Matrix:")
sim_df = pd.DataFrame(similarity_matrix,
                      columns=[f'Doc {i+1}' for i in range(len(docs))],
                      index=[f'Doc {i+1}' for i in range(len(docs))])
print(sim_df.round(2))

# 6. Heatmap for Similarity
plt.figure(figsize=(6, 5))
sns.heatmap(sim_df, annot=True, cmap="Oranges", fmt=".2f")
plt.title("Document Similarity Matrix (BoW)")
plt.show()
