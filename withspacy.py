import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Natural Language Processing with spaCy is really interesting and powerful!"

# Process the text
doc = nlp(text)

# Tokenization: List all tokens
tokens = [token.text for token in doc]
print("Tokens:", tokens)


