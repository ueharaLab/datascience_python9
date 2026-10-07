import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from pylab import rcParams
import japanize_matplotlib
import pandas as pd

cookpad_df = pd.read_csv('dataset_tf_cookpad.csv', encoding='ms932', sep=',')
labels = cookpad_df['label_digits'].values
features = cookpad_df.iloc[:,3:].values
feature_names = cookpad_df.iloc[:,3:].columns.tolist()
label_names = cookpad_df['label'].values.tolist()

X_train,X_test,Y_train,Y_test = train_test_split(features,labels,random_state=3)


#予測精度が最大になるtree depthで学習
tree = DecisionTreeClassifier(max_depth=5,random_state=0)
tree.fit(X_train,Y_train)

training_score = tree.score(X_train,Y_train)
print('training score:', training_score)
test_score = tree.score(X_test,Y_test)
print('test score', test_score)

feature_dict={}
for feature_label, feature_figure in zip(feature_names,tree.feature_importances_):
	if feature_figure != 0.:
		print(feature_label)
		feature_dict[feature_label]=feature_figure
feature_sorted = sorted(feature_dict.items(), key=lambda x:x[1],reverse=False)

feature_name=[]
feature_importances_nonzero=[]
for k,v in feature_sorted:
	feature_name.append(k)
	feature_importances_nonzero.append(v)
		
plt.barh(range(len(feature_name)),feature_importances_nonzero,height=0.5)
plt.yticks(np.arange(len(feature_name)),feature_name,fontsize=8)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()




rcParams['figure.figsize'] = 15,8
plot_tree(tree,feature_names=feature_names,filled=True, rounded=True, fontsize=10)
plt.show()



