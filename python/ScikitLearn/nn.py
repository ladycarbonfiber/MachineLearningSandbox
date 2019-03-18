import util
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

c, f, t_C, t_f = util.load_letter_data_set()
scaler = preprocessing.StandardScaler().fit(f)
f = scaler.transform(f)
t_f = scaler.transform(t_f)
classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,20), random_state=1)
classifier.fit(f,c)
predictions = classifier.predict(t_f)
error_rate = len(t_C[predictions!=t_C])/len(t_C)
training_predictions = classifier.predict(f)
training_error_rate = len(c[training_predictions!=c])/len(c)
print("letter training error rate {}".format(training_error_rate))
print("letter error rate {}".format(error_rate))

c, f, t_C, t_f = util.load_census_data_set()
scaler = preprocessing.StandardScaler().fit(f)
f = scaler.transform(f)
t_f = scaler.transform(t_f)
classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)
classifier.fit(f,c)
predictions = classifier.predict(t_f)
error_rate = len(t_C[predictions!=t_C])/len(t_C)
training_predictions = classifier.predict(f)
training_error_rate = len(c[training_predictions!=c])/len(c)
print("Income training error rate {}".format(training_error_rate))
print("Income error rate {}".format(error_rate))
