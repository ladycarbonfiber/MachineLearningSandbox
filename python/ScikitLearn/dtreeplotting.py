import util
import string
import matplotlib.pyplot as plt
import numpy
from sklearn import tree

k=2
cc, ff, t_cc, t_ff = util.load_census_data_set()
clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=5).fit(ff,cc)
bf, btf = util.best_k_features(ff,cc,t_ff, k)
clf.fit(bf,cc)
predictions = clf.predict(btf)
error_rate = len(t_cc[predictions!=t_cc])/len(t_cc)
training_predictions = clf.predict(bf)
training_error_rate = len(cc[training_predictions!=cc])/len(cc)
print("income 2 feature training error rate {}".format(training_error_rate))
print("income 2 feature error rate {}".format(error_rate))

#Decision surfice plots of 2 "best" features
#code impsitred from the scikit-learn documentation
if k==2:
    feature_names = ['age', 'hours worked']
    labels = ['<=50k', '>50K']
    plot_colors = "by"
    plot_step = .02
    x_min, x_max = bf[:,0].min() -1, bf[:,0].max()+1
    y_min, y_max = bf[:,1].min() -1, bf[:,1].max()+1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, plot_step),
                            numpy.arange(y_min, y_max, plot_step))    
    Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.axis("tight")
    for i, color in zip(range(2), plot_colors):
        idx = numpy.where(cc == i)
        plt.scatter(bf[idx, 0], bf[idx, 1], c=color, label=labels[i],
                    cmap=plt.cm.Paired)
    plt.title("Decision surface of best two income featuers")
    plt.legend()
    fig1 = plt.gcf()
    fig1.savefig('decisionSurface.png')