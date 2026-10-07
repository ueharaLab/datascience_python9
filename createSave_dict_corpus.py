import gensim
from gensim import corpora, models
from collections import defaultdict
import pandas as pd
import numpy as np
import keitaiso_mecab

def create_dict(csv_name):
	csv_input = pd.read_csv(csv_name, encoding='ms932', sep=',')
	review_text_list=[]
	for words in csv_input['ingredients']:
	
		sozai_list=keitaiso_mecab.tokenizer(words)
		review_text_list.append(sozai_list.split(','))
# ------------------辞書生成---------------------------------------
#文書語彙の行列を渡すだけで、辞書が生成される（辞書は語彙のリスト形式）
#辞書はgensim Dictonaryクラスのオブジェクト
	dictionary = gensim.corpora.Dictionary(review_text_list)
	dictionary.filter_extremes(no_below=20, no_above=0.3)
	dictionary.save_as_text('cookpad.dict')
	print(dictionary)
# ----------------コーパス生成
#上記辞書を参照しながら、形態素解析済の記事を文書単位毎にid（辞書のインデックス）を付与し、出現頻度を計算する
#corpusは、(id,頻度) id順に並べなおされる
#.doc2bowメソッドは、文書1単位の語彙リストから、辞書を参照してコーパスを生成するもの
#これを文書数分生成するので、リスト内包表記で行列として生成する

	corpus = [dictionary.doc2bow(review_morph) for review_morph in review_text_list]
	print(corpus[0])
	print(review_text_list[0])

	gensim.corpora.MmCorpus.serialize('cookpad_corpus.mm', corpus) 
	return dictionary,corpus