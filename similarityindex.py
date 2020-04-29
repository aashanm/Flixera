# Program for quantifying similarity between 2 pieces of text
# Serves as basic foundation for MLfrom sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample strings to compare
text = ["London Paris London", "Paris Paris London"]
cv = CountVectorizer()

# vectorize text based on word frequency
count_matrix = cv.fit_transform(text)

# calculate angle between 2 vectors
similarity_scores = cosine_similarity(count_matrix)
print (similarity_scores)