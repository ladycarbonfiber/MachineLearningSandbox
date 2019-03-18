#Gradient Tree Boosting
import util
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

c, f, t_C, t_f = util.load_letter_data_set()
classifier = RandomForestClassifier(n_estimators=5)
classifier.fit(f,c)
predictions = classifier.predict(t_f)
error_rate = len(t_C[predictions!=t_C])/len(t_C)
training_predictions = classifier.predict(f)
training_error_rate = len(c[training_predictions!=c])/len(c)
print("letter training error rate {}".format(training_error_rate))
print("letter error rate {}".format(error_rate))

c, f, t_C, t_f = util.load_census_data_set()
#classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=.75, max_depth=4, random_state=0)
classifier.fit(f,c)
predictions = classifier.predict(t_f)
error_rate = len(t_C[predictions!=t_C])/len(t_C)
training_predictions = classifier.predict(f)
training_error_rate = len(c[training_predictions!=c])/len(c)
print("Income training error rate (RandomForestClassifier){}".format(training_error_rate))
print("Income error rate (RandomForestClassifier) {}".format(error_rate))

c, f, t_C, t_f = util.load_census_data_set()
classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=.75, max_depth=4, random_state=0)
classifier.fit(f,c)
predictions = classifier.predict(t_f)
error_rate = len(t_C[predictions!=t_C])/len(t_C)
training_predictions = classifier.predict(f)
training_error_rate = len(c[training_predictions!=c])/len(c)
print("Income training error rate (GradientBoostingClassifier) {}".format(training_error_rate))
print("Income error rate (GradientBoostingClassifier) {}".format(error_rate))