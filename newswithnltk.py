import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK data files (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Input text
text = ("External Affairs Minister S Jaishankar urged the global community to understand that "
        "the recent confrontation between India and Pakistan was not just a conflict between two neighbours, "
        "but it was about combating terrorism, which, he said, will eventually come back to haunt the West.")

# Step 1: Tokenization
tokens = word_tokenize(text)
print("1️⃣ Tokens:\n", tokens)

# Step 2: Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\n2️⃣ After Stopword Removal:\n", filtered_tokens)

# Step 3: Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_tokens]
print("\n3️⃣ After Stemming:\n", stemmed)

# Step 4: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\n4️⃣ After Lemmatization:\n", lemmatized)
