from lcs import longest_common_subsequence

class LCSSetAlgorithm():

	"""
	The LeastCommonSequence algorithm is a modification of the Least Common Subsequence (LCS) problem we face in
	finding common subsequences between two different strings of different lengths.

	The class is an implementation of the LCS algorithm combined with a modified version of LCS, known as
	a normalized LCS which takes in the string inputs and finds the minimum pair distance between two least common
	subsequences.

	Parameters:

	1. text_1: str
		Takes the first string file the user submits.

	2. text_2: str
		Takes the second string file the user submits. The similarity of the text will be compared against this given
		text file.

	Methods:

	1. __intersect: Private class method used to find the intersection of two given strings. The function
	is not directly called but used as a subroutine in maximal_consecutive_lcs and maximal_consecutive_lcsN.

		Parameters:
			string1: The string that is extracted from text_1

			string2: The string that is extracted from text_2

		Output:
			result: Returns the variable 'result' that holds the intersection of the words.

	2. __maximal_consecutive_lcs: Private class method that detects the maximal consecutive least
												common subsequence between two strings.

		Parameters:
			t1_len: The length of the first text string "text_1".

			t2_len: The length of the second text string "text_2".

		Output:
			len_subset_str: Returns the minimum length of the LCS subset detected between the two strings that are
			passed into the method.

	3. normalized_lcs: Class method that computes the normalized LCS score of the two text strings that are submitted.

		Output:
			NCLS: The normalized least common subsequence score

			The output "NCLS" is computed using the following formula:

			(lcs_len)^2 / length of text 1 * length of text 2

	"""
	def __init__(self, text_1, text_2):
		self.text_1 = text_1
		self.text_2 = text_2
		self.lcs_len = longest_common_subsequence(self.text_1, self.text_2)

		if type(text_1) is not str:
			raise ValueError("The text document supplied is not a valid string input!. Please supply a text string")

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
		t2_len = len(self.text_2)

		if t1_len > t2_len:
			self.text_1 = self.text_2
			self.text_2 = self.text_1
		else:
			pass

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