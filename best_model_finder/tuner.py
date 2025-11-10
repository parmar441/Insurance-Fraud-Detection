from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.sv_classifier=SVC(probability=True)
        self.xgb = XGBClassifier(
            objective='binary:logistic',
            n_jobs=-1,
            random_state=42,
            eval_metric='logloss',
            tree_method='hist'
        )

    def get_best_params_for_svm(self,train_x,train_y):
        """
        Method Name: get_best_params_for_naive_bayes
        Description: get the parameters for the SVM Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_svm method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {
                "kernel": ['rbf', 'poly'],
                "C": [0.1, 1.0, 10.0],
                "gamma": ['scale', 'auto']
            }

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(
                estimator=self.sv_classifier,
                param_grid=self.param_grid,
                cv=5,
                verbose=0,
                n_jobs=-1
            )
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.kernel = self.grid.best_params_['kernel']
            self.C = self.grid.best_params_['C']
            self.gamma = self.grid.best_params_['gamma']


            #creating a new model with the best parameters
            self.sv_classifier = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True)
            # training the new model
            self.sv_classifier.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'SVM best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_svm method of the Model_Finder class')

            return self.sv_classifier
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_svm method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'SVM training  failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {
                "n_estimators": [200, 400],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(
                XGBClassifier(
                    objective='binary:logistic',
                    n_jobs=-1,
                    random_state=42,
                    eval_metric='logloss',
                    tree_method='hist'
                ),
                self.param_grid_xgboost,
                verbose=0,
                cv=3,
                n_jobs=-1
            )
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.subsample = self.grid.best_params_['subsample']
            self.colsample_bytree = self.grid.best_params_['colsample_bytree']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(
                objective='binary:logistic',
                n_jobs=-1,
                random_state=42,
                eval_metric='logloss',
                tree_method='hist',
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree
            )
            # training the new model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.prediction_xgboost = self.xgboost.predict(test_x)
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.prediction_xgboost = self.xgboost.predict_proba(test_x)[:,1]
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC

            # create best model for Random Forest
            self.svm=self.get_best_params_for_svm(train_x,train_y)
            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.prediction_svm=self.svm.predict(test_x) # prediction using the SVM Algorithm
                self.svm_score = accuracy_score(test_y,self.prediction_svm)
                self.logger_object.log(self.file_object, 'Accuracy for SVM:' + str(self.svm_score))
            else:
                self.prediction_svm=self.svm.predict_proba(test_x)[:,1]
                self.svm_score = roc_auc_score(test_y, self.prediction_svm) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for SVM:' + str(self.svm_score))

            #comparing the two models
            if(self.svm_score <  self.xgboost_score):
                return 'XGBoost',self.xgboost
            else:
                return 'SVM',self.sv_classifier

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

