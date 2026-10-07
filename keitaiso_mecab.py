import re
import sys
import MeCab


def tokenizer(sentence):
	japanese = re.compile("[一-龥ぁ-んァ-ンー。、々]")
	sentence_list = re.findall(japanese,sentence)
	sentence = ''.join(sentence_list)

	t = MeCab.Tagger ('mecabrc')
	t.parse('')	
		
	#t.parseToNode('') 
	word_vector = ''	
	token=t.parseToNode(sentence)	
	while token:
		if token.feature.split(',')[0] == '名詞':
			
			word_vector += token.surface+ ','
			
		token = token.next
	return word_vector[:-1]
	
	
