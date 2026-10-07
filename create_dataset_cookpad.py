import pandas as pd
import createSave_dict_corpus
import create_dataset_tf_idf
import codecs
import numpy as np

csv_name='pancake_hotcake.csv'

dictionary,corpus = createSave_dict_corpus.create_dict(csv_name)
feature_matrix,feature_matrix_tfidf,dict_word=create_dataset_tf_idf.making_dataset(dictionary,corpus)

csv_input = pd.read_csv(csv_name, encoding='ms932', sep=',')
dataset_tf_df=pd.DataFrame(feature_matrix,columns=dict_word)
label_digits = np.zeros(len(feature_matrix))
label_digits[csv_input['label']=='ホットケーキ']=1
label_digits_df = pd.DataFrame(label_digits,columns=['label_digits'])
dataset_tf_df = pd.concat([label_digits_df,csv_input['label'],csv_input['title'],dataset_tf_df],axis=1)
with codecs.open("dataset_tf_cookpad.csv", "w", "ms932", "ignore") as dataset:
		
	dataset_tf_df.to_csv(dataset, index=False, encoding="ms932", mode='w', header=True)


dataset_tfidf_df=pd.DataFrame(feature_matrix_tfidf,columns=dict_word)
dataset_tfidf_df = pd.concat([label_digits_df,csv_input['label'],csv_input['title'],dataset_tfidf_df],axis=1)
with codecs.open("dataset_tf_idf_cookpad.csv", "w", "ms932", "ignore") as dataset2:
		
	dataset_tfidf_df.to_csv(dataset2, index=False, encoding="ms932", mode='w', header=True)









