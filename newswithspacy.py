import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Input text
text = ("External Affairs Minister S Jaishankar urged the global community to understand that "
        "the recent confrontation between India and Pakistan was not just a conflict between two neighbours, "
        "but it was about combating terrorism, which, he said, will eventually come back to haunt the West.")

# Process the text
doc = nlp(text)

# 1️⃣ Tokenization
tokens = [token.text for token in doc]
print("1️⃣ Tokens:\n", tokens)

# 2️⃣ Stopword Removal
filtered_tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
print("\n2️⃣ After Stopword Removal:\n", filtered_tokens)

# 3️⃣ Lemmatization
lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
print("\n3️⃣ After Lemmatization:\n", lemmatized_words)
