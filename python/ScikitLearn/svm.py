#SVM analysis using sklearn
import util
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
c, f, t_C, t_f = util.load_letter_data_set()
#suppert vector machines are not scale invarient
scaler = preprocessing.StandardScaler().fit(f)
f = scaler.transform(f)
t_f = scaler.transform(t_f)

linear_classifier = svm.SVC(kernel='linear',cache_size=1000).fit(f,c)
predictions_l = linear_classifier.predict(t_f)
error_rate_l = len(t_C[predictions_l!=t_C])/len(t_C)
print("letter error rate (linear){}".format(error_rate_l))
poly_classifier_2 = svm.SVC(kernel='poly', degree=2,cache_size=1000).fit(f,c)
predictions_p_2 = poly_classifier_2.predict(t_f)
error_rate_p_2 = len(t_C[predictions_p_2!=t_C])/len(t_C)
print("letter error rate (poly_ord_2){}".format(error_rate_p_2))
poly_classifier = svm.SVC(kernel='poly', degree=3,cache_size=1000).fit(f,c)
predictions_p = poly_classifier.predict(t_f)
error_rate_p = len(t_C[predictions_p!=t_C])/len(t_C)
print("letter error rate (poly_ord_3){}".format(error_rate_p))




c, f, t_C, t_f = util.load_census_data_set()
scaler = preprocessing.StandardScaler().fit(f)
f = scaler.transform(f)
t_f = scaler.transform(t_f)
linear_classifier = svm.SVC(kernel='linear',cache_size=1000).fit(f,c)
predictions_l = linear_classifier.predict(t_f)
error_rate_l = len(t_C[predictions_l!=t_C])/len(t_C)
print("income error rate (linear){}".format(error_rate_l))
poly_classifier_2 = svm.SVC(kernel='poly', degree=2,cache_size=1000).fit(f,c)
predictions_p_2 = poly_classifier_2.predict(t_f)
error_rate_p_2 = len(t_C[predictions_p_2!=t_C])/len(t_C)
print("income error rate (poly_ord_2){}".format(error_rate_p_2))
poly_classifier = svm.SVC(kernel='poly', degree=3,cache_size=1000).fit(f,c)
predictions_p = poly_classifier.predict(t_f)
error_rate_p = len(t_C[predictions_p!=t_C])/len(t_C)
print("income error rate (poly_ord_3){}".format(error_rate_p))
