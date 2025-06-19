from textblob import TextBlob
from langdetect import detect

# Input text
text = "I realy love this movie! It was awsome and full of emotion."

# Create a TextBlob object
blob = TextBlob(text)

# Show original text
print("Original Text:", text)

# 1. Correct Spelling
print("Corrected Text:", blob.correct())

# 2. Sentiment Analysis
print("Sentiment (polarity, subjectivity):", blob.sentiment)

# 3. Language Detection
print("Language Detected:", detect(text))