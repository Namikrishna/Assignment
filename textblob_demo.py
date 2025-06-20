from textblob import TextBlob
from langdetect import detect

text = "I realy love this movie! It was awsome and full of emotion."

blob = TextBlob(text)

print("Original Text:", text)
print("Corrected Text:", blob.correct())

# Sentiment analysis in separate lines
print("Sentiment Analysis:")
print("  ➤ Polarity    :", blob.sentiment.polarity)
print("  ➤ Subjectivity:", blob.sentiment.subjectivity)

# Language detection using langdetect
print("Language Detected:", detect(text))