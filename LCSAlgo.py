from lcs import longest_common_subsequence

class LCSSetAlgorithm():

	def __init__(self, text_1, text_2, threshold_check=int):
		self.text_1 = text_1
		self.text_2 = text_2
		self.threshold_check = threshold_check
		self.lcs_len = longest_common_subsequence(self.text_1, self.text_2)

	def __intersect(self, string1, string2):
		result = ''
		# finding the common characters from both strings
		for chars in string1:
			if chars in string2 and not chars in result:
				result += chars
		return result

	# Write a driver code that makes the smaller string "t1" and the longer string "t2".
	def __maximal_consecutive_lcs(self):
		t1_len = len(self.text_1)

		while t1_len > 0:
			intersection = self.__intersect(self.text_1, self.text_2)
			if intersection in self.text_2:
				len_subset_str = len(intersection)
				return len_subset_str
			else:
				self.text_1 = self.text_1[:-1]
				t1_len = len(self.text_1)

	def __maximal_consecutive_lcsN(self):
		intersection_length = self.__intersect(self.text1, self.text2)


	def normalized_lcs(self):
		NLCS = (self.lcs_len**2)/(len(self.text_1) * len(self.text_2))
		return NLCS

if __name__ == '__main__':
	sentence = "Maybe Adam needs to understand that he is the bad guy!"
	sentence_2 = "Maybe Adam should understand that he is in the wrong?"
	plag = LCSSetAlgorithm(sentence, sentence_2)
	plag.normalized_lcs()

###################################################################################

# t1 = "It will be alright"
# t2 = "It will be alrite"
#
#
# def intersect(string1, string2):
#     result = ''
#     # finding the common chars from both strings
#     for chars in string1:
#         if chars in string2 and not chars in result:
#             result += chars
#     return result
#
#
# def mclcs(text1, text2):
# 	len_text1 = len(text1)
# 	len_text2 = len(text2)
#
# 	while len_text1 > 0:
# 		intersection = intersect(text1, text2)
# 		if intersection in text2:
# 			len_subset_str = len(intersection)
# 			print(len_subset_str)
# 			break
# 		else:
# 			text1 = text1[:-1]
# 			len_text1 = len(text1)
#
# mclcs(t1,t2)