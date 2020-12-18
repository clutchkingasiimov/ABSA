import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from LCSAlgo import LCSSetAlgorithm
from jaccardsim import JaccardCheck
import nltk 
nltk.download('punkt')
nltk.download('stopwords')

class Plagiarism():

	"""
	The Plagiarism class is the defining class that is used for detecting the text similarity between two given text
	inputs. The Plagiarism class is used in conjuction with the LCSSetAlgorithm module in order to compute the LCS set
	and feed it further alongside TF-IDF in order to calculate the text similarity between two given input strings.

	The module utilizes stopwords for the english language since the text detection is done on strings written in the
	english language.

	Parameters:

		Global Parameters:
		stopwords: The corpus of stopwords that are to be used for the processing of text. Is shared with all class
		methods.

		Instance parameters

		1. text1: The first text input by the user

		2. text2: The second text input by the user

		3. texts: List created of the two text inputs given by the user (Implicitly defined within the class)

	Methods:

		__text_tokens: Creates tokens of the sentence, which is used for calculating the TF-IDF
		transformation of the text strings.

			Parameters:
				str_to_tokenize: str
				Takes in the string that has to be tokenized and stored for the TF-IDF transformation

			Output:
				tokens: Returns the tokenized string

		__stopwords_remover: Removes the stopwords from the text string. This is one of the key requirements for running
		the TF-IDF algorithm.

			Parameters:
				str_to_tokenize: str
				Takes in the string that has to be tokenized and stored for the TF-IDF transformation

				Utilizes the __text_tokens() private class to first tokenize the strings and then remove the
				stop words

			Output:
				new_text: The clean processed text that is free of stopwords.

		__remove_characters: Removes special characters from the string. This is one of the key requirements for running
		the TF-IDF algorithm.

			Parameters:
				str_to_tokenize: str
				Takes in the string that has to be tokenized and stored for the TF-IDF transformation

				Utilizes the __text_tokens() private class to first tokenize the strings and then remove the
				stop words

			Output:
				clean_text: The final processed dataset which is clean after stopword removal and special character
				removal.


		tf_idf_transform: The TF-IDF transformation method that converts the cleaned processed text into a pairwise
		similarity matrix using TF-IDF.

		The pairwise similarity matrix is computed by multiplying the TF-IDF matrix with its transpose in order to
		calculate the pairwise matrix.

			Parameters:
				clean_text1: The first cleaned text that has been preprocessed by __remove_characters and
				__stopwords_remover.

				clean_text2: The first cleaned text that has been preprocessed by __remove_characters and
				__stopwords_remover.

			Output:

				(pairwise_similarity).A)[0,1]: Returns the pairwise similarity matrix, with its first column that holds
				the similarity vector.

	"""

	#Load all the stopwords in the english language
	stopwords = nltk.corpus.stopwords.words('english')

	def __init__(self, text1, text2):
		self.text1 = text1
		self.text2 = text2
		self.texts = [self.text1, self.text2]

	#Creates tokens of the sentence (Usage is for jaccard similarity or raw Tf-Idf checks)
	def __text_tokens(self, str_to_tokenize):
		tokens = nltk.word_tokenize(str_to_tokenize) #Tokenizes the sentence
		tokens = [token.strip().lower() for token in tokens] #Stores them in a list and makes them all lowercase
		return tokens

	#Removes all the stopwords (Usage is optional)
	def __stopwords_remover(self, str_to_tokenize):
		sentence_tokens = self.__text_tokens(str_to_tokenize) #Tokenize the sentence
		token_filters = [token for token in sentence_tokens if token not in self.stopwords] #Keep all the tokens that are not stopwords
		new_text = ' '.join(token_filters) #Join the processed sentence back
		return new_text

	#Removes special characters like ".", "!", "$", etc
	def __remove_characters(self, str_to_tokenize):
		sentence_tokens = self.__text_tokens(str_to_tokenize) #First we create the tokens of the sentence
		special_character_vals = re.compile('[{}]'.format(re.escape(string.punctuation))) #Keeping the special characters, we re-format the punctuation
		token_filters = filter(None, [special_character_vals.sub(' ', token) for token in sentence_tokens])
		clean_text = ' '.join(token_filters)
		return clean_text

	#Creates a bag of words where each word is tracked by its count
	# def bag_of_words(self, clean_text1, clean_text2):
	# 	word_dict = gensim.corpora.Dictionary(clean_text1) #First store the words in a corpora dictionary
	# 	word_corpus = [] #Create the word-corpus
	# 	sim_check = [] #Create a similarity check list
	# 	#With a for-loop, count the frequency in which each word appears in the sentences and store them in "word_corpus"
	# 	for text in clean_text1:
	# 		word_corpus.append(word_dict.doc2bow(text))
	#

	def tf_idf_transformation(self, clean_text1, clean_text2):
		tfidf = TfidfVectorizer()
		self.X = tfidf.fit_transform([clean_text1, clean_text2])
		pairwise_similarity = self.X * self.X.T

		return ((pairwise_similarity).A)[0,1]

	#Private function for calculating the Jaccard score
	def jaccard_similarity(self):
		jaccard = JaccardCheck(self.text1, self.text2)
		two_word_jaccard, three_word_jaccard, k_word_jaccard = jaccard.jaccard_score_estimation()
		return two_word_jaccard, three_word_jaccard, k_word_jaccard


	def compute_pairwise_similarity(self):
		#First, process the strings for mishaps and stopwords
		processed = self.__remove_characters(self.text1)
		processed_2 = self.__remove_characters(self.text2)
		processed_stop_removed = self.__stopwords_remover(processed)
		processed_stop_removed_2 = self.__stopwords_remover(processed_2)

		#Calculate the pairwise metric score using TF-IDF
		pairwise_metric = self.tf_idf_transformation(processed, processed_2)

		#Implement the LCSSetAlgorithm on the processed strings
		lcs_algo = LCSSetAlgorithm(processed_stop_removed, processed_stop_removed_2)
		lcs_pairwise_similarity = lcs_algo.normalized_lcs()

		two_word_jaccard, three_word_jaccard, k_word_jaccard = self.jaccard_similarity()
		avg_jaccard_score = (two_word_jaccard + three_word_jaccard + k_word_jaccard) / 3
		#Compute the pairwise similarity score as the weighted average of the TF-IDF score and LCSSet score.
		pairwise_score = 100 * ((pairwise_metric) + (lcs_pairwise_similarity) + avg_jaccard_score) / 3


		print("Sentence 1: {}".format(self.text1))
		print("Sentence 2: {}".format(self.text2))
		print("Text Similarity Detected: {}%".format(np.round(pairwise_score)))
		print("Lexical Similiarity Detected (2-Shingle Jaccard Score): {}%".format(np.round(two_word_jaccard * 100)))
		print("Lexical Similiarity Detected (3-Shingle Jaccard Score): {}%".format(np.round(three_word_jaccard * 100)))
		# print("Lexical Similiarity Detected (K-Shingle Jaccard Score): {}%".format(np.round(k_word_jaccard * 100)))


######################################################3
sentence = "Hello world my name is Apple"
sentence_2 = "Hello world my name is Apple"
detection = Plagiarism(sentence, sentence_2)
sim_check = detection.compute_pairwise_similarity()