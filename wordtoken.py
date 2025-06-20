import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

# Sample text
text = "Natural Language Processing with NLTK is really interesting and powerful!"

# Tokenize the text
tokens = word_tokenize(text)
print("Tokens:", tokens)
