import re
from collections import OrderedDict

def exact_consonant_match_compute(string1, string2):
	print(string1,string2)

	vowels = ['a','e','i','o','u']

	for letter in vowels:
		string1 = string1.replace(letter,'')
		string2 = string2.replace(letter,'')

	print(string1,string2)
	score = 0

	# if strings only contain one vowel which has been removed leaving no characters at all
	if string1 == '' or string2 == '':
		return score

	# if first string is shorter
	if len(string1) < len(string2):

		matches = 0
		diff = len(string2) - len(string1)
		for i in range(len(string1)):
			if string1[i] == string2[i]:
				matches += 1
		if matches == 0:
			return 0
		score = matches / (matches+diff)


	elif len(string1) > len(string2):
		matches = 0
		diff = len(string1) - len(string2)
		for i in range(len(string2)):
			if string1[i] == string2[i]:
				matches += 1
		if matches == 0:
			return 0
		score = matches / (matches+diff)

	elif len(string1) == len(string2):
		matches = 0
		diff = len(string1) - len(string2)
		for i in range(len(string2)):
			if string1[i] == string2[i]:
				matches += 1
		if matches == 0:
			return 0
		score = matches / matches

	return score

# file adresses of two languages
lang1 = "cebuano_words.txt" 
lang2 = "bikol_words.txt"

lang1_words = []
lang2_words = []

with open(lang1, "r") as file1:
	lang1_words = file1.readlines()

	for i in range(len(lang1_words)):
		lang1_words[i] = lang1_words[i].lower().strip()

	print(lang1_words)

with open(lang2, "r") as file2:
	lang2_words = file2.readlines()

	for i in range(len(lang2_words)):
		lang2_words[i] = lang2_words[i].lower().strip()

	print(lang2_words)


total_points = 0
genetic_distance = 0

for word1,word2 in zip(lang1_words,lang2_words):

	acc_score = exact_consonant_match_compute(word1,word2)
	print(acc_score)
	total_points += (acc_score*100)

temp = total_points/100
genetic_distance = 100-temp

print("Total Points:",total_points)
print("Genetic Distance:",genetic_distance)

