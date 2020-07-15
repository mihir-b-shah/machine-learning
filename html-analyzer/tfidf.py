from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'The cat is big',
    'The dog is fat'
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

print(X)
