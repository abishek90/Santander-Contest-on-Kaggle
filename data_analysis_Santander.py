# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:30:05 2016

@author: abishek
"""

# This is a script to do exploratory data analysis.

import pandas as pd
import csv
import numpy as np

# The Machine Learning imports
from sklearn.preprocessing import StandardScaler # we will use this to standardize the data
from sklearn.metrics import roc_auc_score # the metric we will be tested on . You can find more here :  https://www.kaggle.com/wiki/AreaUnderCurve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold # the cross validation method we are going to use
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier

SEED = 12
def count_unique(rownum,raw_matrix):
    dict_var = {}
    for vals in raw_matrix[:,rownum]:
        if vals in dict_var:
            dict_var[vals] += 1
        else:
            dict_var[vals] = 1
    return dict_var


def XGB_Holdout(training_data,training_labels,testing_data,testing_labels,true_test_matrix):
    
    scaler=StandardScaler()    

    
    #Throwing the ID away
    training_data_nid = training_data[:,1:]  
    testing_data_nid = testing_data[:,1:]
    true_test_nid = true_test_matrix[:,1:]

   

    total_train_data = np.insert(training_data_nid,training_data_nid.shape[0],testing_data_nid,axis=0) # This is to do the final training in which we submit the model
    total_train_labels = np.insert(training_labels,training_labels.shape[0],testing_labels,axis=0)
    training_labels_total = np.insert(training_labels,training_labels.shape[0],testing_labels,axis=0)

    total_train_data = scaler.fit_transform(total_train_data)
    training_data_nid = scaler.transform(training_data_nid)
    testing_data_nid = scaler.transform(testing_data_nid)
    true_test_nid = scaler.transform(true_test_nid)
#    

    
    
    # ONE HOT ENCODING - TURNED OUT TO BE USELESS

   
    
    param_test1 = {'max_depth':range(1,15,10),'n_estimators' : [100,120,200,300,140,100,320,350,280,400], 'colsample_bytree' : [0.4,0.75,0.65,0.8,0.5,0.3,0.55,0.25,0.35]}
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11), param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(training_data_nid,training_labels)
    bp = gsearch1.best_params_
    print bp
    gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(training_data_nid, training_labels)
    #gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=3,min_child_weight=1, gamma=0, subsample=0.4,objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(training_data_nid, training_labels)
    
    predictions = gbm.predict_proba(testing_data_nid)[:,1]
    
    # Compare the AUC of predictions with testing_labels
    roc_auc = roc_auc_score(testing_labels,predictions)
    print "AUC : %f" % (roc_auc)

    # Hyperparameter Tuning For the total train data.

    param_test1 = {'max_depth':range(1,15,10),'n_estimators' : [100,120,200,300,140,100,320,350,280,400], 'colsample_bytree' : [0.4,0.75,0.65,0.8,0.5,0.3,0.55,0.25,0.35]}
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11), param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(total_train_data,training_labels_total)
    bp = gsearch1.best_params_
    print bp
    
    gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(total_train_data, training_labels_total)
    
    true_preds = gbm.predict_proba(true_test_nid)[:,1]
    return true_preds
    


def ExtraTrees(training_data,training_labels,testing_data,testing_labels,true_test_matrix):
    # This is based on the classifier of Extremely Random Trees

     #Throwing the ID away
    training_data_nid = training_data[:,1:]  
    testing_data_nid = testing_data[:,1:]
    true_test_nid = true_test_matrix[:,1:]
    
    total_train_data = np.insert(training_data_nid,training_data_nid.shape[0],testing_data_nid,axis=0)
    training_labels_total = np.insert(training_labels,training_labels.shape[0],testing_labels,axis=0)
    

    param_test1 = {'n_estimators':[200,300,400],'criterion' : ['gini','entropy'], 'bootstrap' : [True, False]}
    gsearch1 = GridSearchCV(estimator = ExtraTreesClassifier(criterion = 'gini',random_state = 11, n_estimators = 20, bootstrap = False) , param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(training_data_nid,training_labels)
    bp = gsearch1.best_params_
    print bp
    ET = ExtraTreesClassifier(criterion = bp['criterion'],random_state = 11, n_estimators = bp['n_estimators'], bootstrap = bp['bootstrap']).fit(training_data_nid,training_labels)
    predictions = ET.predict_proba(testing_data_nid)[:,1]
    
    roc_auc = roc_auc_score(testing_labels,predictions)
    print "AUC : %f" % (roc_auc)
    
    ET = ExtraTreesClassifier(criterion = bp['criterion'],random_state = 11, n_estimators = bp['n_estimators'], bootstrap = bp['bootstrap']).fit(total_train_data,training_labels_total)
    true_preds = ET.predict_proba(true_test_nid)[:,1]
    return true_preds


def XGB_kfoldstrat(training_data,training_labels,testing_data,testing_labels,true_test_matrix):
    # This function will fit a XGB model on train_data and then see its performance on testing_labels
    
    scaler=StandardScaler()    
    X_train = scaler.fit_transform(training_data)
    X_cv = scaler.fit_transform(testing_data)
    
    overall_train = np.insert(X_train,X_train.shape[0],X_cv,axis=0)
    overall_train2 = np.insert(training_data,training_data.shape[0],testing_data,axis=0)
    
    using_train = overall_train2
    print using_train.shape
    y = training_labels
    print y
    kfolder=StratifiedKFold(y, n_folds= 5,shuffle=True, random_state=SEED) 
    i=0
    mean_auc = 0.0
    for train_index, test_index in kfolder:
        X_train, X_cv = using_train[train_index], using_train[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
        param_test1 = {'max_depth':range(3,13,4),'n_estimators' : [200,300,140,100], 'colsample_bytree' : [0.4,0.8]}
        gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11), param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
        gsearch1.fit(X_train,y_train)
        bp = gsearch1.best_params_
        print bp
        gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(X_train, y_train)
        #gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=3,min_child_weight=1, gamma=0, subsample=0.75,objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(training_data, training_labels)
    
        predictions = gbm.predict_proba(X_cv)[:,1]
    
        # Compare the AUC of predictions with testing_labels
        roc_auc = roc_auc_score(y_cv,predictions)
        print "AUC : %f" % (roc_auc)
        mean_auc += roc_auc
        i+=1
    print "Mean AUC : %f" %(mean_auc/5)
    
    true_preds = gbm.predict_proba(true_test_matrix)[:,1]
    return true_preds
    

    
def logreg_one_hot_encoding(training_data,training_labels,testing_data,testing_labels,true_test_matrix):
    
    # Needs ONC of the total data in training_data + testing_data + true_test_matrix. We will skip using the ID in this example
    training_matrix_nid = training_data[:,1:]
    testing_matrix_nid = testing_data[:,1:]
    true_test_matrix_nid = true_test_matrix[:,1:]
    # The above strips the data of the id which we dont want to onc.
    
    overall_dat1 = np.insert(training_matrix_nid,training_matrix_nid.shape[0],testing_matrix_nid,axis=0)
    print training_matrix_nid.shape    
    overall_dat2 = np.insert(overall_dat1,overall_dat1.shape[0],true_test_matrix_nid,axis=0)
    enc = OneHotEncoder()
    enc.fit(overall_dat2)
    feature_matrix = enc.transform(overall_dat2).toarray()
    
    training_feature = feature_matrix[0:training_matrix_nid.shape[0],:]
    testing_feature = feature_matrix[training_matrix_nid.shape[0]:training_matrix_nid.shape[0]+testing_matrix_nid.shape[0],:]
    true_test_feature = feature_matrix[training_matrix_nid.shape[0]+testing_matrix_nid.shape[0],:]
    
    # Learn a LogReg Model
    param_test1 = {'C':[1,1.6,1.8,1.9,2,3],'penalty' : ['l1']}
    gsearch1 = GridSearchCV(estimator = LogisticRegression(penalty = 'l1', random_state = 11, C = 1) , param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(training_feature,training_labels)
    bp = gsearch1.best_params_
    print bp
    logit = LogisticRegression(penalty = bp['penalty'], random_state = 11, C = bp['C']).fit(training_feature,training_labels)
    predictions = logit.predict_proba(testing_feature)[:,1]
    
    # Compute AUC
    roc_auc = roc_auc_score(testing_labels,predictions)
    print "AUC : %f" % (roc_auc)
    true_preds = logit.predict_proba(true_test_feature)[:,1]
    return true_preds
    
    



def Ensembling2_RF_XGB_Submission(training_data,training_labels,testing_data,testing_labels,true_test_matrix):
    # Testing data is to generate a CV score for this method.
    training_data_nid = training_data[:,1:]  
    testing_data_nid = testing_data[:,1:]
    true_test_nid = true_test_matrix[:,1:]


    scaler = StandardScaler()
   

    total_train_data = np.insert(training_data_nid,training_data_nid.shape[0],testing_data_nid,axis=0) # This is to do the final training in which we submit the model
    total_train_labels = np.insert(training_labels,training_labels.shape[0],testing_labels,axis=0)
    
    total_train_data = scaler.fit_transform(total_train_data)
    training_data_nid = scaler.transform(training_data_nid)
    testing_data_nid = scaler.transform(testing_data_nid)
    true_test_nid = scaler.transform(true_test_nid)

    training_data = training_data_nid
    testing_data = testing_data_nid
    true_test_matrix = true_test_nid
    
    print training_data_nid.shape

    total_train_data = np.insert(training_data_nid,training_data_nid.shape[0],testing_data_nid,axis=0) # This is to do the final training in which we submit the model
    total_train_labels = np.insert(training_labels,training_labels.shape[0],testing_labels,axis=0)

    int_split = 0.5

    n_rows_train = training_data.shape[0]

    train_data_A = training_data[0:int(np.floor(n_rows_train*int_split)),:]
    train_data_B = training_data[int(np.floor(n_rows_train*int_split)):,:]

    train_labels_A = training_labels[0:int(np.floor(n_rows_train*int_split))]
    train_labels_B = training_labels[int(np.floor(n_rows_train*int_split)):]

    # Train XGB on A and predict on B and vice-versa
    curr_data = train_data_A
    curr_labels = train_labels_A

    to_pred_data = train_data_B


    param_test1_XGB = {'max_depth':range(2,8,3),'n_estimators' : [200,300,150], 'colsample_bytree' : [0.4,0.8]}
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11), param_grid = param_test1_XGB, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(curr_data,curr_labels)
    bp_XGB = gsearch1.best_params_
    print bp_XGB
    gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=bp_XGB['n_estimators'], max_depth=bp_XGB['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp_XGB['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(curr_data, curr_labels)
    XGB_preds_B = gbm.predict_proba(to_pred_data)[:,1]
    print np.array([XGB_preds_B]).shape

    curr_data = train_data_B
    curr_labels = train_labels_B

    to_pred_data = train_data_A

    param_test1_XGB = {'max_depth':range(2,8,3),'n_estimators' : [100,200,300], 'colsample_bytree' : [0.4,0.8]}
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11), param_grid = param_test1_XGB, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(curr_data,curr_labels)
    bp_XGB = gsearch1.best_params_
    print bp_XGB
    gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=bp_XGB['n_estimators'], max_depth=bp_XGB['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp_XGB['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(curr_data, curr_labels)
    XGB_preds_A = gbm.predict_proba(to_pred_data)[:,1]

    #XGB_pred_train = np.insert(np.array([XGB_preds_A]), np.array([XGB_preds_A]).shape[1],np.array([XGB_preds_B]) ,axis=0)
    XGB_pred_train = np.hstack(( np.array([XGB_preds_A]) , np.array([XGB_preds_B]) ) )
    print np.transpose(XGB_pred_train).shape

    
    train_stack_dummy = np.hstack(   ( training_data_nid, (np.matrix(XGB_pred_train).T)   ) )  

    # Train RandomForest on A and predict on B and vice-versa.

    curr_data = train_data_A
    curr_labels = train_labels_A

    to_pred_data = train_data_B

    param_testRF = {'n_estimators':[300,400], 'max_depth' : [10,None], 'criterion' : ['entropy']}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(random_state = SEED, n_estimators = 10, max_depth = 1, criterion = 'entropy') , param_grid = param_testRF, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(curr_data,curr_labels)
    bp_RF = gsearch1.best_params_
    curr_model = RandomForestClassifier(n_estimators=bp_RF['n_estimators'], criterion = 'entropy', max_depth=bp_RF['max_depth'],min_samples_leaf=1,max_features=0.4,n_jobs=3,random_state=SEED)
    rfc = curr_model.fit(curr_data,curr_labels)
    pred_rfc_B = curr_model.predict_proba(to_pred_data)[:,1]
    

    curr_data = train_data_B
    curr_labels = train_labels_B

    to_pred_data = train_data_A


    param_testRF = {'n_estimators':[300,400], 'max_depth' : [10, None], 'criterion' : ['entropy']}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(random_state = SEED, n_estimators = 10, max_depth = 1, criterion = 'entropy') , param_grid = param_testRF, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(curr_data,curr_labels)
    bp_RF = gsearch1.best_params_
    curr_model = RandomForestClassifier(n_estimators=bp_RF['n_estimators'], criterion = 'entropy', max_depth=bp_RF['max_depth'],min_samples_leaf=1,max_features=0.4,n_jobs=3,random_state=SEED)
    rfc = curr_model.fit(curr_data,curr_labels)
    pred_rfc_A = curr_model.predict_proba(to_pred_data)[:,1]

    #RF_pred_train = np.insert(np.array([pred_rfc_A]), np.array([pred_rfc_A]).shape[1], np.array([pred_rfc_B]),axis=0)
    RF_pred_train = np.hstack( (np.array([pred_rfc_A]) , np.array([pred_rfc_B])) )
    print RF_pred_train.shape
    # RF_pred_train and XGB_pred_train are the two feature columns over which we need to train another XGb.

    train_stack = np.hstack(   ( training_data_nid, (np.matrix(XGB_pred_train).T)   , (np.matrix(RF_pred_train).T)  ) )  
    # Train a XGB Combiner

    param_test1_XGB = {'max_depth':range(2,8,3),'n_estimators' : [100,200], 'colsample_bytree' : [0.4,0.8]}
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11), param_grid = param_test1_XGB, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(train_stack,training_labels)
    bp_XGB = gsearch1.best_params_
    print bp_XGB
    gbmComb = xgb.XGBClassifier( learning_rate =0.05, n_estimators=bp_XGB['n_estimators'], max_depth=bp_XGB['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp_XGB['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(train_stack, training_labels)
    XGB_preds_A = gbmComb.predict_proba(to_pred_data)[:,1]

    # CV Score Estimation - 

    # Training XGB on Entire Training Set

    param_test1_XGB = {'max_depth':range(1,15,5),'n_estimators' : [100,200,300], 'colsample_bytree' : [0.4,0.75,0.8,0.5]}
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11), param_grid = param_test1_XGB, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(training_data,training_labels)
    bp_XGB = gsearch1.best_params_
    print bp_XGB
    gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=bp_XGB['n_estimators'], max_depth=bp_XGB['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp_XGB['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(training_data, training_labels)
    XGB_preds_test = gbm.predict_proba(testing_data_nid)[:,1]

    # Training the RF on the Entire Training  Set

    param_testRF = {'n_estimators':[350,400], 'max_depth' : range(1,10,3), 'criterion' : ['entropy']}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(random_state = SEED, n_estimators = 10, max_depth = 1, criterion = 'entropy') , param_grid = param_testRF, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(training_data_nid,training_labels)
    bp_RF = gsearch1.best_params_
    curr_model = RandomForestClassifier(n_estimators=bp_RF['n_estimators'], criterion = 'entropy', max_depth=bp_RF['max_depth'],min_samples_leaf=1,max_features=0.4,n_jobs=3,random_state=SEED)
    rfc = curr_model.fit(training_data_nid,training_labels)
    pred_rfc_test = curr_model.predict_proba(testing_data_nid)[:,1]

    test_stack = np.hstack(   ( testing_data_nid, (np.matrix([XGB_preds_test]).T)  , (np.matrix([pred_rfc_test]).T) ) ) 

    # Combining using gbmComb

    combiner_preds = gbmComb.predict_proba(test_stack)[:,1]
    # See the CV- Meaningless
    roc_auc = roc_auc_score(testing_labels,combiner_preds)
    print "AUC : %f" % (roc_auc)
    return roc_auc

    # Generate the Submission File - The Entire Training Set Step is skipped

    # XGB on the whole data 



def RandomForest(training_data,training_labels,testing_data,testing_labels,true_test_matrix):
    training_data_nid = training_data[:,1:]  
    testing_data_nid = testing_data[:,1:]
    true_test_nid = true_test_matrix[:,1:]
    
    scaler = StandardScaler()
   

    total_train_data = np.insert(training_data_nid,training_data_nid.shape[0],testing_data_nid,axis=0) # This is to do the final training in which we submit the model
    total_train_labels = np.insert(training_labels,training_labels.shape[0],testing_labels,axis=0)
    
    total_train_data = scaler.fit_transform(total_train_data)
    training_data_nid = scaler.transform(training_data_nid)
    testing_data_nid = scaler.transform(testing_data_nid)
    true_test_nid = scaler.transform(true_test_nid)
    # Train the RF 



    param_testRF = {'n_estimators':[350,400], 'max_depth' : range(1,10,3), 'criterion' : ['entropy']}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(random_state = SEED, n_estimators = 10, max_depth = 1, criterion = 'entropy') , param_grid = param_testRF, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(training_data_nid,training_labels)
    bp_RF = gsearch1.best_params_

#    clf = RandomForestClassifier()
#    param_test_RF = {"max_depth": range(1,10,4),
#              "max_features": [0.4],
#              "min_samples_split": [1, 3, 10],
#              "min_samples_leaf": [1],
#              "bootstrap": [True, False],
#              "criterion": [ "entropy"],
#                "n_estimators": [100,200]}
#                
#    gsearch1 = GridSearchCV(estimator =  RandomForestClassifier()  , param_grid = param_test_RF, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    print bp_RF
    curr_model = RandomForestClassifier(n_estimators=bp_RF['n_estimators'], criterion = 'entropy', max_depth=bp_RF['max_depth'],min_samples_leaf=1,max_features=0.4,n_jobs=3,random_state=SEED)
    rfc = curr_model.fit(training_data_nid,training_labels)
    pred_rfc = curr_model.predict_proba(testing_data_nid)[:,1]
    roc_auc = roc_auc_score(testing_labels,pred_rfc)
    print "AUC : %f" % (roc_auc)


def make_submission(row_id,predictions,filename):
    with open(filename, 'w') as f:
        f.write("id,Response\n")
        for i, labels in enumerate(predictions):
            f.write("%d,%f\n" % (row_id[i], labels))

    
def main():
    raw_data = pd.read_csv('train.csv',delimiter=',')
    raw_matrix = raw_data.as_matrix()
    print raw_matrix[0,0]
    print len(raw_matrix[1,:])
    # 370 Features in the data

    
    test_split_ratio = 0.75;
    n_rows_raw_data = raw_matrix.shape[0]
    
    nfeatures = 369;
    training_data = raw_matrix[0:int(np.floor(n_rows_raw_data*test_split_ratio)),0:nfeatures+1]
#    
    training_labels = raw_matrix[0:int(np.floor(n_rows_raw_data*test_split_ratio)),nfeatures+1]
#    
    testing_data = raw_matrix[int(np.floor(n_rows_raw_data*test_split_ratio)+1):,0:nfeatures+1]
    testing_labels = raw_matrix[int(np.floor(n_rows_raw_data*test_split_ratio)+1):,nfeatures+1]
#   
    
    # Generate a submission file 
    
    true_test_raw = pd.read_csv('test.csv',delimiter=',')
    true_test_matrix = true_test_raw.as_matrix();
    #true_test_matrix = true_test_matrix
    
    #XGB
    preds = XGB_Holdout(training_data,training_labels,testing_data,testing_labels,true_test_matrix)
    make_submission(true_test_matrix[:,0],preds,"XGB1.csv")
   

    #Random Forest
    #preds = RandomForest(training_data,training_labels,testing_data,testing_labels,true_test_matrix)
    
    
    
    #print len(raw_matrix[:,rownum])
    
if __name__ == '__main__':
    main()
    
    

    