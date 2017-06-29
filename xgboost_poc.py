import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import random

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')



def generate_dataset():
    with open('train.csv', 'w') as file:
        file.write("video_id,section_id,timestamp,movement,label\n")        
        for i in xrange(7):
            for j in xrange(16):
                for k in xrange(0, 1000, 1):
                    movment_feature = 0.050847458 + random.uniform(-0.05,0.05)
                    hour = random.randrange(1,24,1)

                    file.write("{},{},{},{},{}\n".format(i,j, hour,movment_feature, 0))

                for k in xrange(0, 1000, 1):
                    movment_feature = 0.50847458 + random.uniform(-0.05,0.4)
                    hour = random.randrange(12,17,1)

                    file.write("{},{},{},{},{}\n".format(i,j, hour,movment_feature, 0))

                for k in xrange(0, 100, 1):
                    movment_feature = 0.491525424 + random.uniform(-0.05, 0.05)
                    hour = random.randrange(1,4,1)

                    file.write("{},{},{},{},{}\n".format(i,j, hour,movment_feature, 1))

    with open('test.csv', 'w') as file:
        file.write("video_id,section_id,timestamp,movement,label\n")        
        for i in xrange(7):
            for j in xrange(16):
                for k in xrange(0, 1000, 1):
                    movment_feature = random.uniform(0,0.4)
                    hour = random.randrange(1,24,1)

                    file.write("{},{},{},{},{}\n".format(i,j, hour,movment_feature, 0))

                for k in xrange(0, 10, 1):
                    movment_feature = random.uniform(0.6,1)
                    hour = random.randrange(1,24,1)

                    file.write("{},{},{},{},{}\n".format(i,j, hour,movment_feature, 1))

generate_dataset()

def modelfit(alg, dtrain,predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain["label"].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['label'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    # print "Model Report"
    # print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions)
    # print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label'], dtrain_predprob)
                    
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')


csv_data_train = pd.read_csv("train.csv")
csv_data_test = pd.read_csv("output.csv")

# csv_data_test = pd.read_csv("test.csv")

predictors = [x for x in csv_data_train.columns if x not in ['label']]

videos_id = set(csv_data_train["video_id"])
sections_id = set(csv_data_train["section_id"])

classifiers = {}

for video_id in videos_id:
    for section_id in sections_id:
        classifier_name = str(video_id) + '_' + str(section_id)
        
        classifiers[classifier_name] = XGBClassifier(
                                                     learning_rate =0.1,
                                                     n_estimators=1000,
                                                     max_depth=5,
                                                     min_child_weight=1,
                                                     gamma=0,
                                                     subsample=0.8,
                                                     colsample_bytree=0.8,
                                                     objective= 'binary:logistic',
                                                     nthread=1,
                                                     scale_pos_weight=1,
                                                     seed=27)

        modelfit(   classifiers[classifier_name], 
                    csv_data_train.loc[(csv_data_train["video_id"]==video_id) & (csv_data_train["section_id"]==section_id)], 
                    predictors)

for video_id in videos_id:
    for section_id in sections_id:
        if video_id != 1 or section_id != 6:
            continue
        classifier_name = str(video_id) + '_' + str(section_id)
        dtest = csv_data_test.loc[(csv_data_test["video_id"]==video_id) & (csv_data_test["section_id"]==section_id)]
        dtest_predictions = classifiers[classifier_name].predict(dtest[predictors])
        dtest_predprob = classifiers[classifier_name].predict_proba(dtest[predictors])[:,1]
        
        print(dtest_predictions)
        print(dtest['label'].values)

        #Print model report:
        print classifier_name
        print "Model Report"

        print "Accuracy : %.4g" % metrics.accuracy_score(dtest['label'].values, dtest_predictions)
        # print "AUC Score (Train): %f" % metrics.roc_auc_score(dtest['label'], dtest_predprob)