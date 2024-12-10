# Cell 

from hyperopt import *

import pickle
import time
from turtle import TPen
import numpy as np
import pandas as pd
import os
import csv
import statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier



from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneGroupOut

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import GroupKFold

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

import os
os.environ['PYSPARK_PYTHON'] = '/s/babbage/b/nobackup/nblancha/merry/conda/envs/BenProj/bin/python'
os.environ['PYSPARK_DRIVER'] = '/s/babbage/b/nobackup/nblancha/merry/conda/envs/BenProj/bin/python'


# Cell 2

pd.set_option('display.max_columns', None)

# Cell 3

#"C:\Users\bpw10\Downloads\Whisper_NonHuman\Whisper_NonHuman\whisper_final.csv"

hunting_set_file = hunting_set_file = r"Verbal_Oracle.csv"

#"C:/Users/bpw10/Downloads/oracle_flattened.csv"
df_hunting = pd.read_csv(hunting_set_file)
print(df_hunting.dtypes)

# # Cell 4

#New code
X = df_hunting.iloc[:, 21:]
print(X.columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#End of new code


# X = df_hunting.iloc[:, 21:]
# print(X.columns)
# X = X.to_numpy()

# Cell 5

y = df_hunting.iloc[:, 2:21]
print(y.columns)
y = y.to_numpy()

# Cell 6

y_facets = []
for i in range(len(y)):
    const = 0
    neg = 0
    maintain = 0
    if(1 in y[i][:5]):
        const = 1
    if(1 in y[i][5:12]):
        neg = 1
    if(1 in y[i][12:]):
        maintain = 1
    y_facets.append([const,neg,maintain])
y = np.array(y_facets)
print(y[:18])

# Cell 7
utteranceIDs = pd.DataFrame(df_hunting['utteranceID'], columns=['utteranceID'])
utteranceIDs = utteranceIDs.to_numpy()
utteranceIDs=utteranceIDs.reshape(-1)

# Cell 8
groups = pd.DataFrame(df_hunting['Group'], columns=['Group'])
groups = groups.to_numpy()
groups = groups.reshape(-1)
print(groups)

# Cell 9
participants = np.unique(groups)
print(participants)
print(len(participants))

# Cell 10
print (X.shape, y.shape)
is_binary = True
# y_binary = []
# for i in range(len(y)):
#     label = [0,0,0]
#     if(y[i] > 0):
#         label[y[i]-1] = 1
#     y_binary.append(label)
# y_binary = np.array(y_binary)
# if(is_binary):
#     y = y_binary
# print(y)

# Cell 11

output_csv = "Google_MultiModal_Results"
# Change each time data is run (not manually automatic just switch name)
results_path = r"Output"
# Creat this locally manually in colab each time
best_results_csv = os.path.join(results_path, ".csv")
modelcsv = os.path.join(results_path, output_csv)
if not os.path.exists(modelcsv):
        os.mkdir(modelcsv)
else:
        print("WARNING: DIRECTORY ALREADY EXISTS")

# Cell 12
avg = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
results = []
group_kfold = GroupKFold(n_splits=len(participants))
logo = LeaveOneGroupOut()
cv = logo#.split(X, y, groups=groups) #group_kfold #

#avg = [1.0,2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
avg = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
print (X.shape, y.shape)

best = 0

final_score = {
               'model_name':[],
               'hyper_paramaters':[],
               'f1_0':[],
               'acc_overall': [],
               'acc_balanced':[],
               'kappa':[],
               'roc':[],
               'roc_w':[],
               'cm': [],
               'roc_avg':[]}

def hyperopt_train_test(params, groups, cv):
    global avg
    global all_results
    global final_score
    model_name = params['type']
    predictioncsv = model_name
    del params['type']
    for item in params.values():
        predictioncsv += '_' + str(item)
    predictioncsv += '.csv'

    if model_name == 'naive_bayes':
        clf = GaussianNB(**params)
    elif model_name == 'AdaBoost':
        clf = AdaBoostClassifier(**params)
    elif model_name == 'LinSVC':
        clf = LinearSVC(**params)
    elif model_name == 'decision_tree':
        clf = DecisionTreeClassifier(**params, random_state=1) #(**params, n_jobs=6)
    elif model_name == 'randomforest':
        clf = RandomForestClassifier(**params, random_state=1) #(**params, n_jobs=6)
    elif model_name == 'knn':
        clf = KNeighborsClassifier(**params)
    elif model_name == 'logistic_regression':
        if 'penalty' in params and params['penalty'].lower() == 'none':
            params['penalty'] = None
        clf = LogisticRegression(**params, random_state=1)
    elif model_name == 'SVC':
        clf = SVC(**params, probability=True)
    elif model_name == 'MLPClassifier':
        clf = MLPClassifier(**params)
    elif model_name == 'GradientBoosting':
        clf = GradientBoostingClassifier(**params)
    else:
        return 0
    #scores =  cross_validate(clf, X, y,scoring=scoring, groups=groups, cv=cv, n_jobs=20, verbose=200)
    num_classes = 4
    if(is_binary): num_classes = 3
    if(is_binary): clf = MultiOutputClassifier(clf)
    
    y_pred = cross_val_predict(clf, X, y, groups=groups, n_jobs=20,cv=cv, verbose=0, method='predict')
    estimator = 'predict_proba'
    if model_name == 'LinSVC':
        y_score = []
        for num in range(num_classes):
            y_score.append(cross_val_predict(LinearSVC(**params), X, y[:,num], groups=groups, n_jobs=20,cv=cv, verbose=0, method='decision_function'))#'decision_function'
    else:
        y_score = cross_val_predict(clf, X, y, groups=groups, n_jobs=20,cv=cv, verbose=0, method=estimator)
    if(is_binary):
        f1_0 = f1_score(y, y_pred, average='weighted')
    else:
        f1_0 = f1_score(y, y_pred, labels=[0,1,2,3], average='weighted')
    print(f1_0)
    # acc_overall = accuracy_score(y, y_pred)
    # print(acc_overall)
    if(is_binary):
        # acc_balanced = balanced_accuracy_score(y[:,0], y_pred[:,0],)
        # kappa = cohen_kappa_score(y[:,0], y_pred[:,0])
        cm = multilabel_confusion_matrix(y, y_pred)
    else:
        # acc_balanced = balanced_accuracy_score(y, y_pred)
        # kappa = cohen_kappa_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
    ###
    roc_auc_ovr = {}
    #There
    weighted_roc = {}
    matthews_cc = {}
    matthews_list = []
    # roc_avg = 0
    roc_avg_list = []
    kappa_list = []
    # acc_balanced = 0
    acc_balanced_list = []
    # acc_overall = 0
    acc_overall_list = []
    for class_id in range(num_classes):
        if(is_binary and model_name != "LinSVC"):
            prob_tmp = np.array(y_score)[class_id,:,1]
            print(prob_tmp.shape)
            true_max_tmp = y[:, class_id]
            pred_tmp = np.array(y_pred)[:, class_id]
        elif(is_binary and model_name == "LinSVC"):
            prob_tmp = np.array(y_score)[class_id,:]
            print(prob_tmp.shape)
            true_max_tmp = y[:, class_id]
            pred_tmp = np.array(y_pred)[:, class_id]
        else:
            prob_tmp = y_score[:, class_id]
            true_max_tmp = [1 if y_tmp == class_id else 0 for y_tmp in y]
            pred_tmp = [1 if p_tmp == class_id else 0 for p_tmp in y_pred]
        acc_balanced_list.append(balanced_accuracy_score(true_max_tmp, pred_tmp))
        acc_overall_list.append(accuracy_score(true_max_tmp, pred_tmp))
        try:
            roc_auc_ovr[class_id] = roc_auc_score(true_max_tmp, prob_tmp)
            weighted_roc[class_id] = roc_auc_score(true_max_tmp, prob_tmp, average='weighted')
            roc_avg_list.append(weighted_roc[class_id])
        except:
            print(f"ROC_AUC Issue with class {class_id}")
            roc_auc_ovr[class_id] = None
        try:
            kappa_list.append(cohen_kappa_score(true_max_tmp, pred_tmp))
            print('kappa: ', kappa_list)
        except:
            print(f"Kappa issue with class {class_id}")
        matthews_cc[class_id] = matthews_corrcoef(true_max_tmp, pred_tmp)
        matthews_list.append(matthews_cc[class_id])
    if(is_binary):
        matthews_cc[3] = 0
        roc_auc_ovr[3] = 0
    else:
        matthews_cc_3 = matthews_cc[3]
    roc_avg = sum(roc_avg_list)/num_classes
    # here
    roc_stdev = 0
    if(len(roc_avg_list) > 1): statistics.stdev(roc_avg_list)
    kappa = sum(kappa_list)/num_classes
    kappa_stdev = 0
    if(len(kappa_list) > 1): statistics.stdev(kappa_list)
    acc_balanced = sum(acc_balanced_list)/num_classes
    acc_balanced_stdev = statistics.stdev(acc_balanced_list)
    acc_overall = sum(acc_overall_list)/num_classes
    acc_overall_stdev = statistics.stdev(acc_overall_list)
    matthews_avg = sum(matthews_list)/num_classes
    matthews_stdev = statistics.stdev(matthews_list)
    ###

    #print("final_score", final_score)
    avg = [model_name, params, hunting_set_file, predictioncsv, is_binary,
           f1_0, acc_overall, acc_overall_stdev, acc_balanced, acc_balanced_stdev, kappa, kappa_stdev,
           roc_avg, roc_stdev, roc_auc_ovr[0], roc_auc_ovr[1], roc_auc_ovr[2], roc_auc_ovr[3],
           matthews_avg, matthews_stdev, matthews_cc[0], matthews_cc[1], matthews_cc[2], matthews_cc[3],
           cm, len(y_pred), y_pred]


    #return scores['test_acc'].mean()
    #return scores['test_acc_balance'].mean()
    #return scores['test_kappa'].mean()
    #return f1_0


    return avg[10], avg

#define your search space
space = hp.choice('classifier_type', [
    {
        'type': 'AdaBoost',
        'n_estimators': hp.choice('AdaBoost_n_estimators', list(range(30, 100, 11))),
        # 'learning_rate': hp.choice('AdaBoost_learning_rate', list(np.linspace(0.1, 4.1, 50))),
        'learning_rate': hp.choice('AdaBoost_learning_rate', [.001, 0.1, 1.0, 10]),
        'algorithm': hp.choice('AdaBoost_algorithm', ['SAMME', 'SAMME.R'])
    },

    # {
        # 'type': 'MLPClassifier',
        # 'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(100,), (50, 50), (50, 30, 20)]),
        # 'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
        # 'solver': hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
        # 'alpha': hp.loguniform('alpha', np.log(1e-6), np.log(1e-2)),
        # 'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive'])
    # },

    {
    'type': 'GradientBoosting',
    'n_estimators': hp.choice('GBM_n_estimators', list(range(30, 100, 11))),
    'learning_rate': hp.uniform('GBM_learning_rate', 0.01, 0.5),
    'max_depth': hp.choice('GBM_max_depth', [3, 4, 5, 6]),
    'subsample': hp.uniform('GBM_subsample', 0.5, 1),
    'max_features': hp.choice('GBM_max_features', ['sqrt', 'log2']), 
     },


    {
        'type': 'naive_bayes',

    },
    {
        'type': 'logistic_regression',
        'penalty' : hp.choice('logistic_regression_penalty', ['l2', 'none']),
        'solver' : hp.choice('logistic_regression_solver', ['lbfgs', 'newton-cg', 'sag', 'saga'])
    },
    {
        'type': 'LinSVC',
        'C': hp.uniform('LinSVC.C', 0, 3.0),

    },
    {
        'type': 'SVC',
        'C': hp.choice('C', [0.1, 1, 10, 100, 1000]),
        'gamma': hp.choice('gamma', [1, 0.1, 0.01, 0.001, 0.0001]),
        'kernel': hp.choice('kernal', ['rbf', 'sigmoid']),

    },

    {
    'type': 'decision_tree',
    'criterion': hp.choice('decision_tree.criterion', ["gini", "entropy"]),
    'splitter': hp.choice('splitter', ["best", "random"]),
    'max_features': hp.choice('decision_tree.max_features', [None, "sqrt", "log2"]),  # Removed 'auto' from here

    },
    {
        'type': 'randomforest',
        'n_estimators': hp.choice('n_estimators', list(range(20, 170, 16))),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'max_features': hp.choice('max_features', [None, "sqrt", "log2"]),

    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('knn_n_neighbors', list(range(1, 150, 10)))
    }

])
count = 1
all_results =[]


def f(params):
    global best, count, avg, all_results, final_score
    count += 1
    penalty_value = params.get('penalty', None)

    # Check if the penalty_value is 'none' and replace it with None
    if penalty_value is not None and isinstance(penalty_value, str):
        params['penalty'] = penalty_value.lower()

    #all kappas were acc. I want to use kappa value not accuracy
    #acc = hyperopt_train_test(params.copy(), groups, cv)
    guide_metric, scores = hyperopt_train_test(params.copy(), groups, cv)
    # Commenting that out
    # Assuming params is a dictionary containing hyperparameters


    
    #print ('new best:', kap, 'using', params['type'])
    results.append(scores)
    #all_results.append(count)
    #a = [count, kappa, params, avg ]
    #writecsv here
    # with open(os.path.join(modelcsv,scores[3]), 'w', newline = '') as modelCSV:
    #     writer = csv.writer(modelCSV)
    #     print("model: ", scores[3])
    #     writer.writerow(['utteranceID', 'True Label', 'Pred Label'])
    #     for pred in range(len(scores[26])):
    #         writer.writerow([utteranceIDs[pred], y[pred], scores[26][pred]])
    if guide_metric > best:
        #print ('NEW best:', guide_metric, 'using', params['type'])
        #print("f1_0, f1_1, acc_overall , acc_balanced, kappa, roc, cm_0_0, cm_0_1, cm_1_0, cm_1_1")
        print (scores)
        best = guide_metric
        #all_results.append(best)
    else:
        #print ('NOT new best:', guide_metric, 'using', params['type'])
        #print("f1_0, f1_1, acc_overall , acc_balanced, kappa, roc, cm_0_0, cm_0_1, cm_1_0, cm_1_1")
        print (scores)


    #print ('iters:', count, ', kappa:', guide_metric, 'using', params)
    return {'loss': -guide_metric, 'status': STATUS_OK, 'eval_time': time.time(),
            'other_stuff': {'model':avg[0],'f1_0':avg[5] ,'acc_overall': avg[6],
                           'acc_balanced': avg[8],'kappa':avg[10] , 'roc': avg[12],'matthews_cc': avg[18]}
    }

spark_trials = SparkTrials(parallelism=30)
best = fmin(
            f,
            space,
            trials=spark_trials,
            algo=tpe.suggest,
            max_evals= 500,) # was 500, change as see fit depending on training
print ('best:' )
print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227)
df = pd.DataFrame(results, columns=["model_name", "params", "featurecsv", "predictioncsv", "is_binary",
                                    "f1", "acc_overall", "acc_overall_stdev", "acc_balanced", "acc_balanced_stdev", "kappa", "kappa_stdev",
                                    "roc_avg", "rov_stdev", 'roc_0', 'roc_1','roc_2', 'roc_3',
                                    "matthews_cc", "matthews_stdev", "matthews_0", "matthews_1", "matthews_2", "matthews_3",
                                    "cm", "num_samples", "y_pred"])
# [model_name, params, hunting_set_file, predictioncsv, is_binary,
#            f1_0, acc_overall, acc_overall_stdev, acc_balanced, acc_balanced_stdev, kappa, kappa_stdev,
#            roc_avg, roc_stdev, roc_auc_ovr[0], roc_auc_ovr[1], roc_auc_ovr[2], roc_auc_ovr[3],
#            matthews_avg, matthews_stdev, matthews_cc[0], matthews_cc[1], matthews_cc[2], matthews_cc[3],
# #            cm, len(y_pred)]
# df = df.sort_values(by="roc_avg", ascending=False)
# df.to_csv(os.path.join(modelcsv, output_csv) + ".csv")
# best_model = list(df.iloc[0])
# best_model.insert(0, output_csv)

# with open (best_results_csv,'a',newline = '') as best_models:
#     writer = csv.writer(best_models)
#     writer.writerow(best_model)


spark_trials.trials
best_results = []
for trial in spark_trials.trials:
    best_results.append(trial['result']['other_stuff'])
listOfAllModels = []
for eachModel in best_results:
    df = pd.DataFrame([eachModel])
    print(df)
    listOfAllModels.append(df)
concatenated_df = pd.concat(listOfAllModels, ignore_index=True)

concatenated_df.to_csv('results.csv')






