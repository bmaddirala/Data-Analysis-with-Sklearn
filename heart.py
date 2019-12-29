import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import sklearn as sl
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import matplotlib
import matplotlib.pyplot as plt
from pylab import *




def main(file_name):
    #reads in data
    x,y = read_data(file_name)

    #Splitting data into training and test sets
    x_train, x_test, y_train, y_test =train_test_split(x , y , test_size= .25)

    #Printing class distribution of training and test sets
    train_pos_count, train_neg_count = get_class_distrib(y_train)
    test_pos_count, test_neg_count = get_class_distrib(y_test)
    print("training class distrib: {:.2f}, {:.2f}".format(train_pos_count/len(y_train),
                                                          train_neg_count/len(y_train)))
    print("test class distrib: {:.2f}, {:.2f}".format(test_pos_count/len(y_test),
                                                      test_neg_count/len(y_test)))
    #Getting accuracy using decision tree model
    tree_clf = learn_tree(x_train, y_train)
    acc_train = test_model(tree_clf, x_train, y_train)
    acc_test = test_model(tree_clf, x_test, y_test)
    print("decision tree training acc: {:.4f}, test acc: {:.4f}".format(acc_train, acc_test))

    #Printing names of top 5 most important variables
    for feat, score in top_features(tree_clf, x.columns, 5):
        print("{} ({:.4f})".format(feat, score))

    #Prints out test set accuracies for different values of k
    k_vals = [1, 3, 5, 7, 9, 19, 39, 49,109, 200]
    for k in k_vals:
        knn_clf = learn_knn(x_train, y_train, k)
        acc_test = test_model(knn_clf, x_test, y_test)
        print("k-nn {} test acc: {:.4f}".format(k, acc_test))

    #Creates the Age and Cholestrol Scatter Plot
    t = x["chol"]
    s = x["age"]
    plt.scatter(s, t)
    xlabel('Age')
    ylabel('Cholestrol Levels')
    title('Age vs Cholestrol Levels')
    grid(True)
    show()

    #Creates the Age and Max Heart Rate Scatter Plot
    thal = x["thalach"]
    plt.scatter(s, thal)
    xlabel('Age')
    ylabel('Max Heart Rate Levels')
    title('Age vs Max Heart Rate Levels')
    grid(True)
    show()



def read_data(file_name):
    df = pd.read_csv("heart.csv")
    x, y = pd.DataFrame(df.drop(columns=["target", "exang"])), pd.Series(df["target"])
    return x, y  # x is a DataFrame, y is a Series


def get_class_distrib(class_labels):
    pos,neg = 0,0
    for x in class_labels:
        if(x == 0):
            neg = neg + 1
        if(x == 1):
            pos = pos + 1
    return pos, neg  # pos and neg are integer counts


def learn_tree(x, y):
    clf = sl.tree.DecisionTreeClassifier()
    clf = clf.fit(x,y)
    return clf  # clf is a tree classifier object


def test_model(clf, x, y):
    y_predict = clf.predict(x)
    acc = accuracy_score(y,y_predict)
    return acc  # acc is a float


def top_features(clf, col_names, num):
    mapNameImp = dict(zip(clf.feature_importances_, col_names))
    i = 0
    feat_scores = []
    for key in sorted(mapNameImp.keys(), reverse= True):
        if i == num:
            return feat_scores
        else:
            feat_scores.append((mapNameImp[key],key))
        i = i + 1
    return feat_scores # feat_scores is a list of (feature name, float) tuples

def learn_knn(x, y, k):
    clf = KNeighborsClassifier(n_neighbors= k)
    clf = clf.fit(x,y)
    return clf  # clf is a knn classifier object


def learn_knn_standard(x, y, k):
    scalar = StandardScaler()
    x = scalar.fit_transform(x,y)
    clf = learn_knn(x,y,k)
    clf = make_pipeline(scalar, clf)
    return clf  # clf is a pipeline object


def crossval_tree(x, y, folds):
    clf = learn_tree(x,y)
    acc = cross_val_score(clf,x, y, cv=folds)
    return np.mean(acc) # acc is a float


#######
if __name__ == '__main__':
    data_file_name = "heart.csv"
    main(data_file_name)
