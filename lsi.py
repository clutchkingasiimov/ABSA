#Implementing Latent Semantic Indexing
import numpy as np
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

class LSI():

	def __init__(self, document_matrix):
		self.document_matrix = document_matrix


	def get_feature_names(self, make_word_cloud=bool):
		topics = self.document_matrix.get_feature_names()

		if make_word_cloud:
			wordcloud = WordCloud(max_font_size=30).generate(topics)
			plt.figure()
			plt.imshow(wordcloud, interpolation="bilinear")
			plt.axis("off")
			plt.show()
		else:
			pass

	def calculate_lsi(self, return_topic_importance=bool):
		t_SVD = TruncatedSVD(algorithm='randomized', n_iter=100)
		lsi = t_SVD.fit_transform(self.document_matrix)

		if return_topic_importance is True:
			print(t_SVD.components_)
			return lsi
		else:
			return lsi



