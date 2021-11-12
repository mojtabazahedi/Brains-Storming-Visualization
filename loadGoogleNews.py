from gensim.models import Word2Vec, KeyedVectors

model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

#dog = model['dog']
print(model.word_vec['computer':5])
