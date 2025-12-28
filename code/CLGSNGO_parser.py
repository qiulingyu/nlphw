from CLGSNGO import *
import pandas as pd

with open('ceb_all_data.txt', 'r', encoding='utf-8', errors='ignore') as file:
	file_contents = file.readlines()

	label = []
	title = []
	tag_bigram = []
	bik_bigram = []
	ceb_bigram = []
	tag_trigram = []
	bik_trigram = []
	ceb_trigram = []

	for item in file_contents:
		parsed_text = item.split(',',2)
		parsed_text[2] = parsed_text[2].strip()

		print(parsed_text[0], parsed_text[1])

		title.append(parsed_text[0])
		label.append(parsed_text[1])

		tag_output, bik_output, ceb_output = get_bigram_CLGSNGO(parsed_text[2])
		tag_bigram.append(tag_output)
		bik_bigram.append(bik_output)
		ceb_bigram.append(ceb_output)

		tag_output, bik_output, ceb_output = get_trigram_CLGSNGO(parsed_text[2])
		tag_trigram.append(tag_output)
		bik_trigram.append(bik_output)
		ceb_trigram.append(ceb_output)

df = pd.DataFrame(list(zip(title, tag_bigram, bik_bigram, ceb_bigram, tag_trigram, bik_trigram, ceb_trigram, label)),columns=['book_title','tagalog_bigram_sim','bikol_bigram_sim','cebuano_bigram_sim','tagalog_trigram_sim','bikol_trigram_sim','cebuano_trigam_sim', 'grade_level'])

df.to_csv('clgsngo.csv',index=False)

print('\nFEATURE EXTRACTION DONE')