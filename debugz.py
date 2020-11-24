from nltk.tokenize import word_tokenize
from nltk.util import ngrams


# Creates all ngrams of the sentence
def n_grammer(sentence):
	ngram_set = []
	for i in range(len(sentence.split())):
		grams = ngrams(sentence.split(), i + 1)
		for gram in grams:
			ngram_set.append(set(gram))
	return ngram_set


def mclsn(text1, text2):
	len_text1 = len(text1)
	len_text2 = len(text2)
	ngram_hash_len = len(n_grammer(text1))

	while ngram_hash_len >= 0:
		ngram_hash = n_grammer(text1)
		x = max(ngram_hash)
		# Assertion check to ensure that x is a subset of r_i
		#         assert x.issubset(ngram_hash)
		text_set = set(text2.split())
		if x.issubset(text_set):
			return x
			break
		else:
			ngram_hash = ngram_hash.remove(x)
			ngram_hash_len = len(ngram_hash)


text1 = "albastru"
text2 = "alabaster"

mclsn(text1, text2)