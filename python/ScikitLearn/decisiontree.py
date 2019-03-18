import util
import string
import pydotplus
from sklearn import tree
cc, ff, t_cc, t_ff = util.load_census_data_set()

clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=5).fit(ff,cc)
predictions = clf.predict(t_ff)
training_predictions = clf.predict(ff)
training_error_rate = len(cc[training_predictions!=cc])/len(cc)
test_error_rate = len(t_cc[predictions!=t_cc])/len(t_cc)
print("income training error rate {}".format(training_error_rate))
print("income test error rate {}".format(test_error_rate))

class_names = ['<=50K', '>50K']
feature_names = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'weekly-work-hours', 'native country']
dot_data = tree.export_graphviz(clf, class_names=class_names, feature_names=feature_names, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_png("census_d_tree.png") 



c, f, t_C, t_f = util.load_letter_data_set()
classifier = tree.DecisionTreeClassifier(max_depth=26, min_samples_leaf=5)
classifier.fit(f,c)
predictions = classifier.predict(t_f)
error_rate = len(t_C[predictions!=t_C])/len(t_C)
training_predictions = classifier.predict(f)
training_error_rate = len(c[training_predictions!=c])/len(c)
print("letter training error rate {}".format(training_error_rate))
print("letter error rate {}".format(error_rate))

bf, btf = util.best_k_features(f,c,t_f)
classifier.fit(bf,c)
predictions = classifier.predict(btf)
error_rate = len(t_C[predictions!=t_C])/len(t_C)
training_predictions = classifier.predict(bf)
training_error_rate = len(c[training_predictions!=c])/len(c)
print("letter training error rate {}".format(training_error_rate))
print("letter error rate {}".format(error_rate))
letter_features = list(string.ascii_lowercase)
#dot_data = tree.export_graphviz(classifier, out_file=None, class_names=letter_features) 
#graph = pydotplus.graph_from_dot_data(dot_data) 
#graph.write_png("letter_d_tree.png") 