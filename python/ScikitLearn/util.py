#helper functions for reading data
import numpy
from numpy import array, genfromtxt
from sklearn import preprocessing
#generate some nice graphs by selecting best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def load_letter_data_set():
    #return  category, features, define last 4000 as testing
    all_data = genfromtxt('data_sets/letter-recognition.data', delimiter=',', dtype='str')
    label = all_data[:,0]
    features = all_data[:,1:].astype(numpy.float)
    return label[:16000], features[:16000,:], label[:-16000], features[:-16000,:]

def load_census_data_set():
    #return training cat, features and testing cat, features
    label_encoder = preprocessing.LabelEncoder()
    training_data = genfromtxt('data_sets/adult.data', delimiter=',', dtype='str')
    test_data = genfromtxt('data_sets/adult.test', delimiter=',', dtype='str')
    #removing education label since it's already encoded for UserWarning
    #removing capital gains/losses since it makes the set uninteresting
    #fnweights end up reflecting demographic data already present
    training_data = numpy.delete(training_data, [2,3,10,11], axis=1)
    test_data = numpy.delete(test_data, [2,3,11,12], axis=1)
    category = training_data[:,-1]
    category = label_encoder.fit_transform(category)
    features = training_data[:,:-1]
    test_category = test_data[:,-1]
    test_category = label_encoder.fit_transform(test_category)
    test_features = test_data[:,:-1]
    full_set = numpy.append(features,test_features, axis=0)
    discrete_columns=[1,2,3,4,5,6,7,9]
    for col in discrete_columns:
        label_encoder.fit(full_set[:,col])
        features[:,col] = label_encoder.transform(features[:,col])
        test_features[:,col] = label_encoder.transform(test_features[:,col])
    features = features.astype(numpy.int)
    test_features = test_features.astype(numpy.int)
    return category, features, test_category, test_features

def best_k_features(f, c, tf, k=2):
    #returns best k features (evaluated on training set) of training and test sets
    pruner = SelectKBest(chi2, k=k).fit(f,c)
    #print(pruner.get_support())
    best_features = pruner.transform(f)
    best_t_features = pruner.transform(tf)

    return best_features, best_t_features

def write_to_file(title, contents):
    out_file = 'results.json'
    with open(out_file, "a+") as f:
        pass

