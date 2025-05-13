from three_step_decoding import *

tsd = ThreeStepDecoding('lid_models/hinglish',
						htrans='nmt_models/rom2hin.pt',
						etrans='nmt_models/eng2eng.pt')

dataset = []

with open('input.txt') as f:
	for line in f:
		line = line.rstrip()
		line = line.split(' ') # can use a custom tokenizer instead
		dataset.append(line)

with open('output.txt', 'w') as f_w:
	for i in range(len(dataset)):
		if i%10 == 0:
			print(f"{i}/{len(dataset)} completed")
		isSingleWord = False # workaround for single word strings
		if len(dataset[i]) == 1:
			isSingleWord = True
			dataset[i].append('.')
		temp = list(tsd.tag_sent(' '.join(dataset[i])))
		# original word, normalized and transliterated word, language tag
		if isSingleWord:
			temp.pop()
		f_w.write(' '.join([word[1] for word in temp]))
