import numpy as np
import gensim
from gensim import corpora, models,matutils
from collections import defaultdict
from operator import itemgetter
import pandas as pd
import codecs

def making_dataset(dictionary,corpus):

	
	dict_word=[]	
	#dictionary = gensim.corpora.Dictionary.load_from_text(dict_name)
	#print('dictionary:',dictionary)
	#0 paddingコーパス(muliutils)を生成する際に、辞書の語彙idの昇順に自動的に並べ替えたベクトルと
	#なるので、その見出しを作るために辞書の語彙をid順にソートしてリストにする
	for k, v in sorted(dictionary.items()):
		
		dict_word.append(v)
	print(type(dictionary))	
	#corpus = corpora.MmCorpus(corpus_name)
	#コーパス行列を逐次読み出して、0 paddingコーパスに変換(.corpus2denseがそのメソッド）。リスト内包表記で0 paddingコーパスを行列にする
	feature_matrix = [list(matutils.corpus2dense([corpus_unit], num_terms=len(dict_word)).T[0]) for corpus_unit in corpus] 

	tfidf = gensim.models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	#コーパス行列を逐次読み出して、0 paddingコーパスに変換(.corpus2denseがそのメソッド）。リスト内包表記で0 paddingコーパスを行列にする
	feature_matrix_tfidf = [list(matutils.corpus2dense([corpus_unit], num_terms=len(dict_word)).T[0]) for corpus_unit in corpus_tfidf] 

	
	#feature_matrix:0 paddingコーパス特徴量行列をndarrayにしたもの
	#feature_matrix_tfidf:0 paddingコーパス特徴量行列をndarrayにしたもの(tf_idf)
	#array_label_id : 教師ラベルを整数ラベルのndarrayにしたもの
	#dict_word: コーパス特徴量行列の列見出し（つまり語彙）
	#label_dict: キー：教師ラベルテキスト，　値：教師ラベルを整数にしたもの　の辞書
	return feature_matrix,feature_matrix_tfidf,dict_word

