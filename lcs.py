def longest_common_subsequence(t1, t2):
	# Calculate the length of the strings
	m = len(t1)
	n = len(t2)

	# Dynamic programming approach for LCS
	L = [[None] * (n + 1) for i in range(m + 1)]
	for i in range(m + 1):
		for j in range(n + 1):
			if i == 0 or j == 0:
				L[i][j] = 0
			elif t1[i - 1] == t2[j - 1]:
				L[i][j] = L[i - 1][j - 1] + 1
			else:
				L[i][j] = max(L[i - 1][j], L[i][j - 1])

	# L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
	return L[m][n]