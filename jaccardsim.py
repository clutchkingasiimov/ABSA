#Implementing Jaccard Similarity
class JaccardCheck():

	"""
	JaccardCheck class calculates the jaccard similarity of the two sentences that are to be used.
	"""

	def __init__(self, document1, document2):
		self.document1 = document1
		self.document2 = document2

	def __kWordGrams(self, document):

		grams_set = set()
		doc_tokens = document.split()
		for i in range(len(doc_tokens)-1):
			if doc_tokens[i] + ' ' + doc_tokens[i+1] != grams_set:
				grams_set.add(doc_tokens[i] + ' ' + doc_tokens[i+1])

		return grams_set

	def __twoShingleGrams(self, document):
		grams_set = set()
		for i in range(len(document) - 1):
			if document[i] + document[i + 1] not in grams_set:
				grams_set.add(document[i] + document[i + 1])
		return grams_set

	def __threeShingleGrams(self, document):
		grams_set = set()
		# 3-Char gram
		for i in range(len(document) - 2):
			if document[i] + document[i + 1] + document[i + 2] not in grams_set:
				grams_set.add(document[i] + document[i + 1] + document[i + 2])

		return grams_set

	def jaccard_score_estimation(self):
		two_shingles_doc1 = self.__twoShingleGrams(self.document1)
		two_shingles_doc2 = self.__twoShingleGrams(self.document2)

		three_shingles_doc1 = self.__threeShingleGrams(self.document1)
		three_shingles_doc2 = self.__threeShingleGrams(self.document2)

		word_shingling_doc1 = self.__kWordGrams(self.document1)
		word_shingling_doc2 = self.__kWordGrams(self.document2)

		#Jaccard estimation for two-word shingles
		two_shingles_jaccard = len(two_shingles_doc1.intersection(two_shingles_doc2)) / \
		                       len(two_shingles_doc1.union(two_shingles_doc2))
		three_shingles_jaccard = len(three_shingles_doc1.intersection(three_shingles_doc2)) / \
		                         len(three_shingles_doc1.union(three_shingles_doc2))

		word_shingles_jaccard = len(word_shingling_doc1.intersection(word_shingling_doc2)) / \
		                        len(word_shingling_doc1.union(word_shingling_doc2))

		return two_shingles_jaccard, three_shingles_jaccard, word_shingles_jaccard





