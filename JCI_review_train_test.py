
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve, precision_score, accuracy_score
import math
import statistics
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from scipy import stats



if __name__ == '__main__':

    binary_class = False

    data_folder = r"C:\Users\super\Documents\Research\Aghi Lab\JCI_review_exc"
    joined_dict = {}
    for file in os.listdir(data_folder):

        df = pd.read_csv(os.path.join(data_folder,file))
        del df['Unnamed: 0']
        condition = file.rstrip(".csv")
        df.dropna(0, 'any', inplace=True)
        joined_dict[condition] = df
        print('Number of', str(condition), ' cells used for training: {:.2f}'.format(len(df)))

    sum_data = pd.DataFrame()
    for condition in joined_dict:
        bad_strings = ['CAFN','CAFO','THB-1','WI38', '1997S', '2124S', 'LN229', 'T98']
        gbm_strings = [ 'GBM6', 'GBM43', 'U251']
        fibroblast_strings = ['1997T', '2124T']
        astro_strings = ['astrocyte']

        if str(condition) not in bad_strings:

            current_df = joined_dict[condition]
            current_df.dropna(0, 'any', inplace=True)

            if str(condition) in fibroblast_strings:
                current_df['Class'] = 1
            elif str(condition) in gbm_strings:
                current_df['Class'] = 0
            elif str(condition) in astro_strings:
                current_df['Class'] = 2


            if sum_data.empty:
                sum_data = current_df
            else:
                sum_data = sum_data.append(current_df)
        print('Number of', str(condition),' cells used for training: {:.2f}'.format(len(current_df)))
    sum_data = sum_data.reset_index(drop=True)

    sum_data.dropna(0,'any',inplace=True)

    astro_data = sum_data[sum_data['Class'] == 2]
    print(astro_data)


    # @TODO: Modify the code here to do a two-class solver between fibroblasts and astrocytes and then test s and ns.

    if binary_class:
        sum_data = sum_data[sum_data['Class'] > 0]

        print(sum_data)



    y = list(sum_data['Class'])
    x = sum_data.loc[:, sum_data.columns != 'Class']

    print(y)
    print(x)



    X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0,train_size=0.7, stratify=y)

    T = 366 + 499
    GBM = 458 + 803 + 797
    AST = 148

    TOT = T + GBM + AST
    w = {0: int(GBM/TOT)}

    classifier = LogisticRegression(solver='saga',random_state=0, max_iter= 100000, multi_class="multinomial",
                                    class_weight='balanced')

    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # # evaluate the model and collect the scores
    # n_scores = cross_val_score(classifier, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # # report the model performance
    # print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


    classifier.fit(X_train, Y_train)

    df = pd.DataFrame(zip(X_train.columns, np.transpose(classifier.coef_)), columns=['features', 'coef'])

    print(df)

    # Accuracy

    n_classes = 3

    y_true = Y_test
    y_pred = classifier.predict(X_test)

    print(y_true)
    print(y_pred)

    target_names = ['GBM', 'Fibroblasts', 'Astrocytes']
    print(classification_report(y_true, y_pred, target_names=target_names))

    accuracy = accuracy_score(y_true, y_pred)

    print('Accuracy: {:.2f}'.format(accuracy))

    # # Precision
    #
    # precision = precision_score(y_true, y_pred, average=None)
    # print('Precision: {:.2f}'.format(precision[0]))
    # print('Precision: {:.2f}'.format(precision[1]))
    # print('Precision: {:.2f}'.format(precision[2]))
    #
    #
    #
    # # Add sensitivity, specificty, and precision
    #
    #
    # mcm = multilabel_confusion_matrix(y_true, y_pred)
    # tn = mcm[:, 0, 0]
    # tp = mcm[:, 1, 1]
    # fn = mcm[:, 1, 0]
    # fp = mcm[:, 0, 1]
    #
    # # Sensitivity
    # sensitivity = tp / (tp + fn)
    # print('Sensitivity: {:.2f}'.format(sensitivity[1]))
    #
    # # Specificity
    # specificity = tn / (tn + fp)
    # print('Specificity: {:.2f}'.format(specificity[1]))
    #
    # precision = tp / (tp + fp)
    #
    # print('Precision: {:.2f}'.format(precision[1]))


    caf_n_data = pd.DataFrame(joined_dict['CAFN'])

    caf_n_data.dropna(0,'any',inplace=True)

    outcome = classifier.predict(caf_n_data)

    data = pd.DataFrame()
    data["Outcome"] = pd.Series(outcome)

    data["Condition"] = "No"
    print(data)


    print("CAFN (serial tryp)")
    print('CAFN GBM: {:.2f}'.format(np.count_nonzero(outcome == 0) / len(outcome)))
    print('CAFN Fibroblast: {:.2f}'.format(np.count_nonzero(outcome == 1) / len(outcome)))
    print('CAFN Astrocyte: {:.2f}'.format(np.count_nonzero(outcome == 2) / len(outcome)))
    print("")
    caf_o_data = pd.DataFrame(joined_dict['CAFO'])

    caf_o_data.dropna(0,'any',inplace=True)

    outcome = classifier.predict(caf_o_data)

    new_data = pd.DataFrame()
    new_data["Outcome"] = pd.Series(outcome)

    new_data["Condition"] = "Yes"

    data = pd.concat([data, new_data], axis=0)

    print(data)


    print("CAFO (non-serial tryp)")
    print('CAFO GBM: {:.2f}'.format(np.count_nonzero(outcome == 0) / len(outcome)))
    print('CAFO Fibroblast: {:.2f}'.format(np.count_nonzero(outcome == 1) / len(outcome)))
    print('CAFO Astrocyte: {:.2f}'.format(np.count_nonzero(outcome == 2) / len(outcome)))
    print("")
    ln229_data = pd.DataFrame(joined_dict['LN229'])

    ln229_data.dropna(0,'any',inplace=True)

    outcome = classifier.predict(ln229_data)

    print('LN229 GBM: {:.2f}'.format(np.count_nonzero(outcome == 0)/len(outcome)))
    print('LN229 Fibroblast: {:.2f}'.format(np.count_nonzero(outcome == 1)/len(outcome)))
    print('LN229 Astrocyte: {:.2f}'.format(np.count_nonzero(outcome == 2)/len(outcome)))
    print("")

    t98_data = pd.DataFrame(joined_dict['T98'])

    t98_data.dropna(0, 'any', inplace=True)

    outcome = classifier.predict(t98_data)

    print('T98 GBM: {:.2f}'.format(np.count_nonzero(outcome == 0) / len(outcome)))
    print('T98 Fibroblast: {:.2f}'.format(np.count_nonzero(outcome == 1) / len(outcome)))
    print('T98 Astrocyte: {:.2f}'.format(np.count_nonzero(outcome == 2) / len(outcome)))
    print("")


    print("Internal Validation")

    # Internal validation

    fibroblast_data = sum_data[sum_data['Class'] == 1]
    gbm_data = sum_data[sum_data['Class'] == 0]
    astro_data = sum_data[sum_data['Class'] == 2]

    del fibroblast_data["Class"]
    del gbm_data["Class"]
    del astro_data["Class"]

    outcome = classifier.predict(fibroblast_data)
    print('Fibroblast Classification: {:.2f}'.format((outcome == 1).sum()/len(outcome)))

    outcome = classifier.predict(gbm_data)
    print('GBM Classification: {:.2f}'.format((outcome == 0).sum()/len(outcome)))

    outcome = classifier.predict(astro_data)
    print('Astrocyte Classification: {:.2f}'.format((outcome == 2).sum()/ len(outcome)))

    # CHI SQUARE TEST OF INDEPENDENCE

    ct = pd.crosstab(data.Condition, data.Outcome,  margins=True)
    print(ct)
    row_sum = ct.iloc[0:2, 3].values
    print(row_sum)


    obs = np.array([ct.iloc[0][0:3].values,
                    ct.iloc[1][0:3].values])
    print(stats.chi2_contingency(obs)[0:3])