import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Input text
text = "Natural Language Processing with spaCy includes tokenizing, stemming, removing stopwords, and lemmatization."

# Apply spaCy pipeline
doc = nlp(text)

# 1. Tokenization
tokens = [token.text for token in doc]
print("1️⃣ Tokens:", tokens)

# 2. Stopword Removal
filtered_tokens = [token.text for token in doc if not token.is_stop]
print("2️⃣ After Stopword Removal:", filtered_tokens)

# 3. Lemmatization (spaCy doesn't support stemming directly)
lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
print("3️⃣ After Lemmatization:", lemmatized_words)

