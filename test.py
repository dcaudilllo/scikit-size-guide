import sys

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from termcolor import colored as cl

def normalize_dataset(path):
    # Read the dataset
    df = pd.read_csv(path)

    df = df.select_dtypes(exclude=['object'])

    # Classifying jeans sizes
    size1 = df[df['currentSize'] == 1][0:50]
    size2 = df[df['currentSize'] == 2][0:50]
    size3 = df[df['currentSize'] == 3][0:50]
    size4 = df[df['currentSize'] == 4][0:50]
    size5 = df[df['currentSize'] == 5][0:50]
    size6 = df[df['currentSize'] == 6][0:50]
    size7 = df[df['currentSize'] == 7][0:50]
    size8 = df[df['currentSize'] == 8][0:50]
    size9 = df[df['currentSize'] == 9][0:50]
    size10 = df[df['currentSize'] == 10][0:50]
    size11 = df[df['currentSize'] == 11][0:50]
    size12 = df[df['currentSize'] == 12][0:50]
    size13 = df[df['currentSize'] == 13][0:50]
    size14 = df[df['currentSize'] == 14][0:50]
    size15 = df[df['currentSize'] == 15][0:50]
    size16 = df[df['currentSize'] == 16][0:50]

    sns.scatterplot(x=size1['Waist'], y=size1['Hips'], s=150, label='Size 0-6')
    sns.scatterplot(x=size2['Waist'], y=size2['Hips'], s=150, label='Size 0-8')
    sns.scatterplot(x=size3['Waist'], y=size3['Hips'], s=150, label='Size 2-6')
    sns.scatterplot(x=size4['Waist'], y=size4['Hips'], s=150, label='Size 2-8')
    sns.scatterplot(x=size5['Waist'], y=size5['Hips'], s=150, label='Size 2-10')
    sns.scatterplot(x=size6['Waist'], y=size6['Hips'], s=150, label='Size 2-12')
    sns.scatterplot(x=size7['Waist'], y=size7['Hips'], s=150, label='Size 4-10')
    sns.scatterplot(x=size8['Waist'], y=size8['Hips'], s=150, label='Size 4-12')
    sns.scatterplot(x=size9['Waist'], y=size9['Hips'], s=150, label='Size 4-14')
    sns.scatterplot(x=size10['Waist'], y=size10['Hips'], s=150, label='size 6-14')
    sns.scatterplot(x=size11['Waist'], y=size11['Hips'], s=150, label='size 8-14')
    sns.scatterplot(x=size12['Waist'], y=size12['Hips'], s=150, label='size 10-16')
    sns.scatterplot(x=size13['Waist'], y=size13['Hips'], s=150, label='Size 10-18')
    sns.scatterplot(x=size14['Waist'], y=size14['Hips'], s=150, label='Size 12-16')
    sns.scatterplot(x=size15['Waist'], y=size15['Hips'], s=150, label='Size 12-18')
    sns.scatterplot(x=size16['Waist'], y=size16['Hips'], s=150, label='Size 16-20')

    plt.legend(fontsize=14)
    plt.title('Jeans Sapphire', fontsize=16)
    plt.xlabel('Waist', fontsize=14)
    plt.ylabel('Hips', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    X_var = np.asarray(df.drop(['currentSize', 'Leg', 'Shape'], axis=1))
    y_var = np.asarray(df['currentSize'])

    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 4)

    # print(cl('X_train samples : ', attrs=['bold']), X_train[:5])
    # print(cl('X_test samples : ', attrs=['bold']), X_test[:5])
    # print(cl('y_train samples : ', attrs=['bold']), y_train[:5])
    # print(cl('y_test samples : ', attrs=['bold']), y_test[:5])

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    yhat = model.predict(X_test)

    import itertools
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=22)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=13)
        plt.yticks(tick_marks, classes, fontsize=13)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     fontsize=15,
                     color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        plt.ylabel('True label', fontsize=16)
        plt.xlabel('Predicted label', fontsize=16)

    cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])
    np.set_printoptions(precision=2)

    # Plot the confusion matrix

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title='Confusion matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    #
    # # preparing the dataset
    # df = df.select_dtypes(exclude=['object'])
    # df = df.fillna(df.mean())
    #
    #
    #
    # X = df.drop('currentSize', axis=1)
    # y = df['currentSize']
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # svr = SVC(kernel='linear', C=1000)
    #
    # sc = StandardScaler().fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)
    #
    # svr.fit(X_train, y_train)
    # y_test_pred = svr.predict(X_test_std)
    # y_train_pred = svr.predict(X_train_std)
    #
    # svr.predict(X_test)
    #
    # print('y_train (r-squared) ---->', r2_score(y_train, y_train_pred))
    # print('y_test (r-squared) ---->', r2_score(y_test, y_test_pred))
    #
    # plt.figure(figsize=(5, 7))

    #
    # # red line means the actual value
    # ax = sns.distplot(y, hist=False, color="r", label="Actual Value")
    #
    # # blue line means the trained value
    # sns.distplot(y_test_pred, hist=False, color="b", label="Fitted Values", ax=ax)
    #
    # plt.title('Actual vs Fitted Values for Sizes')
    # plt.grid()
    # plt.savefig('myfile.png')
    # # plt.show()
    # plt.close()

normalize_dataset('./saphiro_dataset.csv')