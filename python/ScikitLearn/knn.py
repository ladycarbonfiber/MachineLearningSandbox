#KNN anaysis 
import util
from sklearn.neighbors import KNeighborsClassifier

c, f, t_C, t_f = util.load_letter_data_set()
k=1
classifier = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', weights='distance')
classifier.fit(f,c)
predictions = classifier.predict(t_f)
error_rate = len(t_C[predictions!=t_C])/len(t_C)
training_predictions = classifier.predict(f)
training_error_rate = len(c[training_predictions!=c])/len(c)
print("letter training error rate {}".format(training_error_rate))
print("letter error rate {}".format(error_rate))

c, f, t_C, t_f = util.load_census_data_set()
k=3
classifier = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', weights='distance')
classifier.fit(f,c)
predictions = classifier.predict(t_f)
error_rate = len(t_C[predictions!=t_C])/len(t_C)
training_predictions = classifier.predict(f)
training_error_rate = len(c[training_predictions!=c])/len(c)
print("Income training error rate {}".format(training_error_rate))
print("Income error rate {}".format(error_rate))