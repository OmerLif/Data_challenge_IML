# Packages / libraries
import os  # provides functions for interacting with the operating system
import numpy as np
import pandas as pd
import os
# from IPython.display import Image
import sklearn.metrics
from matplotlib import pyplot as plt
import seaborn as sns
np.set_printoptions(formatter={'float_kind': '{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(8,6)})

# Datetime lib
from pandas import to_datetime
import itertools
import warnings
import datetime
import graphviz

warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score


# from sklearn.model_selection import train_test_split


def forest(file_name):
    raw_data = pd.read_csv(file_name, encoding='latin-1')
    # print(raw_data.shape)
    # print(raw_data.head())
    return raw_data


def Investigate(raw_data):
    # Investigate all the elements whithin each Feature
    for column in raw_data:
        unique_vals = np.unique(raw_data[column])
        nr_values = len(unique_vals)
        if nr_values < 12:
            print('The number of values for feature {} :{} -- {}'.format(column, nr_values, unique_vals))
        else:
            print('The number of values for feature {} :{}'.format(column, nr_values))

    # Checking for null values
    print(raw_data.isnull().sum())


def limit_data(raw_data):
    # Limiting the data
    raw_data2 = raw_data[['CreditScore', 'Geography',
                          'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                          'IsActiveMember', 'EstimatedSalary', 'Exited']]

    new_raw_data = pd.get_dummies(raw_data2, columns=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])
    # print(new_raw_data.columns)
    return new_raw_data


def scale_data(raw_data):
    # Scaling our columns - like average
    scale_vars = ['CreditScore', 'EstimatedSalary', 'Balance', 'Age']
    scaler = MinMaxScaler()
    raw_data[scale_vars] = scaler.fit_transform(raw_data[scale_vars])
    # print(raw_data.head())
    return raw_data


def split_data(new_raw_data):
    # Your code goes here
    X = new_raw_data.drop('Exited', axis=1).values  # Input features (attributes)
    y = new_raw_data['Exited'].values  # Target vector
    # print('X shape: {}'.format(np.shape(X)))
    # print('y shape: {}'.format(np.shape(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)
    return X_train, X_test, y_train, y_test, new_raw_data


def graph(new_raw_data):
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=2,
                                random_state=1)  # choose max_depth, Caution - over fit
    dt.fit(X_train, y_train)
    dot_data = tree.export_graphviz(dt, out_file='x',
                                    feature_names=new_raw_data.drop('Exited', axis=1).columns,
                                    class_names=new_raw_data['Exited'].unique().astype(str),
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data).view()
    # graph.view()
    # graph.render(directory='doctest-output/round-table.gv.pdf',format='jpg', view=True)


def feature_importances(new_raw_data):
    # Calculating FI
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=2,
                                random_state=1)  # choose max_depth, Caution - over fit
    dt.fit(X_train, y_train)
    # for i, column in enumerate(new_raw_data.drop('Exited', axis=1)):
    #     print('Importance of feature {}:, {:.3f}'.format(column, dt.feature_importances_[i]))
    #
    #     fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [dt.feature_importances_[i]]})
    #
    #     try:
    #         final_fi = pd.concat([final_fi, fi], ignore_index=True)
    #     except:
    #         final_fi = fi

    # Ordering the data
    # final_fi = final_fi.sort_values('Feature Importance Score', ascending=False).reset_index()
    # final_fi

def accuracy(X_train, X_test, y_train, y_test, new_raw_data):
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
    dt.fit(X_train, y_train)
    # Accuracy on Train
    # print("Training Accuracy is: ", dt.score(X_train, y_train))

    # Accuracy on Train
    # print("Testing Accuracy is: ", dt.score(X_test, y_test))
    return dt


def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def Plotting_Confusion_Matrix(dt):
    #[[true negativ , false negativ ]
    #[ false positive  true positive]]
    y_pred = dt.predict(X_train)

    # Plotting Confusion Matrix
    cm = confusion_matrix(y_train, y_pred)
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    plt.figure().show()
    plot_confusion_matrix(cm_norm, classes=dt.classes_, title='Training confusion')
    y_pred = dt.predict(X_train)
    # print(confusion_matrix(y_train, y_pred))

def starting_Random_forest(X_train, X_test, y_train, y_test, new_raw_data):
    rf = RandomForestClassifier(n_estimators=100,criterion='entropy')
    rf.fit(X_train, y_train)
    prediction_test = rf.predict(X=X_test)
    # Accuracy on Test
    print("Training Accuracy is: ", rf.score(X_train, y_train))
    # Accuracy on Train
    print("Testing Accuracy is: ", rf.score(X_test, y_test))

    # Confusion Matrix
    cm = confusion_matrix(y_test, prediction_test)
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_norm, classes=rf.classes_)

def Tunning_Random_Forest(X_train, X_test, y_train, y_test, new_raw_data):
    # Tunning Random Forest
    from itertools import product
    n_estimators = 100
    max_features = [1, 'sqrt', 'log2'] #The number of features to consider when looking for the best split
    max_depths = [None, 2, 3, 4, 5]
    for f, d in product(max_features, max_depths):  # with product we can iterate through all possible combinations
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    criterion='entropy',
                                    max_features=f,
                                    max_depth=d,
                                    n_jobs=2,
                                    random_state=1337)
        rf.fit(X_train, y_train)
        prediction_test = rf.predict(X=X_test)
        print('Classification accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d,
                                                                                                             accuracy_score(
                                                                                                                 y_test,
                                                                                                                 prediction_test)))
        cm = confusion_matrix(y_test, prediction_test)
        cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        # plot_confusion_matrix(cm_norm, classes=rf.classes_,
        #                       title='Confusion matrix accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(
        #                           f, d, accuracy_score(y_test, prediction_test)))

def test_data():
    sklearn.metrics.roc_curve()



if __name__ == '__main__':
    raw_data = forest('/Users/orishemer/PycharmProjects/Data_challenge_IML/house_prices2.csv')
    # Investigate(raw_data)
    raw_data = limit_data(raw_data)
    raw_data = scale_data(raw_data)
    X_train, X_test, y_train, y_test, new_raw_data = split_data(raw_data)
    feature_importances(new_raw_data)
    dt = accuracy(X_train, X_test, y_train, y_test, new_raw_data)
    Plotting_Confusion_Matrix(dt)
    # starting_Random_forest(X_train, X_test, y_train, y_test, new_raw_data)
    Tunning_Random_Forest(X_train, X_test, y_train, y_test, new_raw_data)

    # graph(new_raw_data)

