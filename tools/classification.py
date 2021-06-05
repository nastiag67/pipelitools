import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import VotingClassifier

from sklearn import metrics

from . import metrics as m


class Simple:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def classification_models(self,
                              average='binary',
                              randomized_search=False,
                              verbose=0):

        """Runs classification models and chooses the best based on accuracy.

        Parameters:
        ----------
        X_train : DataFrame
            Features used in training.

        y_train : Series
            Labels for training (1D vector).

        X_test : DataFrame
            Features used in testing.

        y_test : Series
            Labels for testing (1D vector).

        average : str, default='binary'
            Used in metrics calculation. If multilabel/multiclass, use one of: [‘micro’, ‘macro’, ‘samples’, ‘weighted’]

        randomized_search : bool, default=False
            Specifies if randomized search of hyperparameters should be done (RandomizedSearchCV).
            By default, uses exhaustive search over the hyperparameter grid (GridSearchCV).

        verbose : bool, default=0
            Verbose the hyperparameter search.

        Returns
        ----------

        best_model : dataframe
        Pipeline of the best model.

        allmodels : dataframe
        All the models used

        """

        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # list of models
        models = [
            ("NB", GaussianNB()),
            ("LR", LogisticRegression(max_iter=10000)),
            ("SVM", SVC()),
            ("KNN", KNeighborsClassifier()),
            ('LDA', LinearDiscriminantAnalysis()),
            ('QDA', QuadraticDiscriminantAnalysis()),
            ("DT", DecisionTreeClassifier()),
            ("MLP", MLPClassifier(random_state=42))
                  ]

        best_accuracy = 0
        best_model = None
        allmodels = {}
        all_accuracy = []
        all_precision = []
        all_recall = []
        all_f1 = []

        for name, model in models:

            # Create the pipeline
            steps = [
                ('scaler', StandardScaler()),
                # ('scaler', RobustScaler()),
                # ('pca', PCA(n_components=)),
                (name, model)
            ]
            pipeline = Pipeline(steps)

            # Specify the hyperparameter space
            if name == 'NB':
                parameters = {}

            elif name == 'LR':
                parameters = {
                    'LR__class_weight': ['balanced', None],
                    'LR__multi_class': ['multinomial', 'auto'],
                    # 'LR__solver': ['saga'],  # supports ‘elasticnet’ penalty, for multiclass problems, DEFAULT=lbfgs
                    # 'LR__penalty': ['elasticnet'],  #default l2
                    # 'LR__l1_ratio': np.arange(0, 1, 0.1)
                }

            elif name == 'SVM':
                parameters = {
                    'SVM__C': [1, 10, 50],
                    # Regularization - tells the SVM optimization how much error is bearable
                    # control the trade-off between decision boundary and misclassification term
                    # smaller value => small-margin hyperplane

                    # 'SVM__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # VERY long pls don't
                    # 'SVM__kernel': ['poly', 'rbf', 'sigmoid'],
                    'SVM__kernel': ['rbf'],

                    'SVM__gamma': [0.1, 0.01]
                    # kernel coefficient
                    # higher value => overfitting
                    }

            elif name == 'KNN':
                parameters = {
                    'KNN__n_neighbors': np.arange(3, 8, 1),
                    'KNN__weights': ['uniform', 'distance'],
                    'KNN__p': [1, 2]
                }

            elif name == 'LDA':
                parameters = {
                    'LDA__n_components': np.arange(2, 7, 1)
                }

            elif name == 'QDA':
                parameters = {
                    'QDA__reg_param': np.arange(0, 1, 0.1)
                }

            elif name == 'DT':
                parameters = {
                    'DT__criterion': ['gini', 'entropy'],
                    # 'DT__max_depth': np.arange(10, 500, 10),
                    'DT__max_depth': np.arange(3, 20, 1),
                    # If max_depth=None => nodes r expanded until all the leaves contain less than min_samples_split samples
                    # 'DT__min_samples_split': [2],
                    # 'DT__min_samples_leaf': [1],
                    # 'DT__min_weight_fraction_leaf': [0],
                    'DT__max_features': ['log2', 'sqrt', 'auto'],
                    # 'DT__max_leaf_nodes': [None],
                    'DT__class_weight': ['balanced', None],
                    'DT__random_state': [42],
                }

            elif name == 'MLP':
                parameters = {
                    'MLP__activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'MLP__solver': ['sgd', 'adam'],
                    'MLP__alpha': [0.01, 0.05, 0.1],
                    'MLP__max_iter': [5000],
                    'MLP__early_stopping': [True],
                    # 'MLP__n_iter_no_change': [10],
                }

            else:
                raise ValueError(f'No such model in the list: {models}')

            print()
            print(f"{10*'='} {name} RESULT {10*'='}")

            if randomized_search:
                # randomised search
                cv = RandomizedSearchCV(estimator=pipeline,
                                        param_distributions=parameters,
                                        cv=5,
                                        # refit='accuracy',
                                        scoring='accuracy',
                                        n_iter=10,
                                        verbose=verbose)

            else:
                # exhaustively consider all parameter combinations (cv=8 fold cross-validation)
                cv = GridSearchCV(estimator=pipeline,
                                  param_grid=parameters,
                                  cv=5,
                                  # refit='accuracy',
                                  scoring='accuracy',
                                  verbose=verbose)

            # Fit to the training set
            cv.fit(self.X_train, self.y_train)

            # Mean cross-validated score of the best_estimator
            print(f"Best score: {round(cv.best_score_, 4)}")

            # Parameter setting that gave the best results on the validation data
            print(f"Tuned parameters: {cv.best_params_}")

            # Predict the labels of the test set
            y_pred = cv.predict(self.X_test)

            # METRICS
            accuracy, precision, recall, f1s = m.classification_metrics(self.X_train, self.y_train, self.y_test,
                                                                        y_pred, cv, average=average)

            # add to the dictionary of models
            allmodels[name] = cv

            # plot the confusion matrix
            m.plot_confusion_matrix(self.y_test.values,
                                    self.y_pred.values,
                                    self.y_test.unique(),
                                    normalize=True,
                                    title=f'Confusion matrix for {name}',
                                    cmap=plt.cm.Blues)

            plt.show()

            all_accuracy.append(accuracy)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1s)

            # choose the best model based on ACCURACY
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = cv
                best_modelname = name
                best_params = cv.best_params_

        print(50 * "_")
        print(f"Best model: {best_modelname}")
        print(f"Tuned parameters: {best_params}")
        print(f"BEST ACCURACY: {round(best_accuracy, 4)}")

        df_scores = dict(zip(allmodels.keys(), all_accuracy))

        result = pd.DataFrame(df_scores, index=[0]).transpose().rename(columns={0: 'Accuracy'})
        result['Precision'] = all_precision
        result['Recall'] = all_recall
        result['F1-score'] = all_f1
        result = result.sort_values('Accuracy', ascending=False)
        print(50 * "-")
        print(round(result, 4))

        return best_model, allmodels

    def best_model(self, name, model, submission=False):

        """Runs trainung pipeline and evaluation of model with given hyperparameters.

        Parameters:
        ----------
        name : str
            Custom name of the model.

        model : pipeline or sklearn classification model.
            Instance of sklearn model

        X_train : DataFrame
            Features used in training.

        y_train : Series
            Labels for training (1D vector).

        X_test : DataFrame, default=None
            Features used in testing.

        y_test : Series, default=None
            Labels for testing (1D vector).

        submission : bool, default=False
            If True, returns the labels trained on X_train, y_train

        Returns
        ----------

        pipeline : dataframe
        Pipeline of the trained model.

        """

        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Create the pipeline
        steps = [
            ('scaler', StandardScaler()),
            # ('pca', PCA()),
            (name, model)
        ]

        pipeline = Pipeline(steps)

        # Fit to the training set
        pipeline.fit(self.X_train, self.y_train)

        if submission:
            return pipeline

        # Predict the labels of the test set
        y_pred = pipeline.predict(self.X_test)

        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        print(f"ACCURACY on train data: {round(pipeline.score(self.X_train, self.y_train), 4)}")
        print(f"ACCURACY on test data: {round(accuracy, 4)}")

        # plot the confusion matrix
        m.plot_confusion_matrix(self.y_test.values,
                                y_pred.values,
                                self.y_test.unique(),
                                normalize=False,
                                title=f'Confusion matrix for {name}',
                                cmap=plt.cm.Blues)

        return pipeline


class Ensemble:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def pipe(self, clf_name, clf,  submission=False):

        # Create the pipeline
        steps = [
            ('scaler', StandardScaler()),
            # ('scaler', RobustScaler()),
            # ('pca', PCA()),
            (clf_name, clf)
        ]
        pipeline = Pipeline(steps)

        # Fit to the training set
        pipeline.fit(self.X_train, self.y_train)

        if submission:
            return pipeline, 0

        # Predict the labels of the test set
        y_pred = pipeline.predict(self.X_test)

        print('-' * 100)
        print(f"CLASSIFIER: {clf_name}")

        m.classification_metrics(self.X_train, self.y_train, self.y_test, y_pred, pipeline)

        return pipeline, y_pred

    def voting(self, classifiers, voting='hard', weights=None, submission=False):

        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        if submission is False:
            # Get the performance of each classifier individually
            for clf_name, clf in classifiers:
                self.pipe(clf_name, clf, submission=submission)

        # Instantiate a VotingClassifier vc
        vc = VotingClassifier(
            estimators=classifiers,
            # If ‘hard’, uses predicted class labels for majority rule voting
            # if ‘soft’, predicts the class label based on the argmax of the sums of the predicted probabilities,
            # which is recommended for an ensemble of well-calibrated classifiers
            voting=voting,
            # Sequence of weights (float or int) to weight the occurrences of predicted class labels (hard voting)
            # or class probabilities before averaging (soft voting). Uses uniform weights if None.
            weights=weights
        )

        model, y_pred = self.pipe('Voting classifier', vc, submission=submission)

        return model

    def stacking_2lvl(self, classifiers, meta_classifier):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        df_pred_train = pd.DataFrame(index=self.X_train.index)
        df_pred_test = pd.DataFrame(index=self.X_test.index)
        # Get the performance of each classifier individually
        for clf_name, clf in classifiers:
            pipeline, y_pred = self.pipe(clf_name, clf, self.X_train, self.y_train, self.X_test, self.y_test)
            df_pred_train[clf_name] = pipeline.predict(self.X_train)
            df_pred_test[clf_name] = y_pred

        # Concatenate X_train with the predictions DataFrame
        X_train_2nd = pd.concat([self.X_train, df_pred_train], axis=1)
        X_test_2nd = pd.concat([self.X_test, df_pred_test], axis=1)

        # don't do this, cuz dataset already preprocessed & scaled
        # model, y_pred = pipe('Two level classifier', meta_classifier, X_train_2nd, y_train, X_test_2nd, y_test)

        # Build the second-layer meta estimator
        clf_stack = meta_classifier
        clf_stack.fit(X_train_2nd, self.y_train)

        # Obtain the final predictions from the second-layer estimator
        y_pred = clf_stack.predict(X_test_2nd)

        print('-' * 100)
        print(f"META-CLASSIFIER:")
        m.classification_metrics(self.X_train, self.y_train, self.y_test, y_pred, pipeline)

        return clf_stack

# # Default function to fit a linear classifier and plot the decision boundary
# def demo_clf(clf, X_train, y_train, name):
#     clf.fit(X_train, y_train)
#     risk_train = 1 - clf.score(X_train, y_train)
#     print('Training risk:', risk_train)
#
#     plt.figure()
#     sns.scatterplot(x=X_train['sepal_length'], y=X_train['sepal_width'], hue=y_train, style=y_train, markers=["o", "s"],
#                     legend=[])
#     x0 = np.arange(4.5, 7, .01)
#     b0 = clf.intercept_
#     b1 = clf.coef_[0, 0]
#     b2 = clf.coef_[0, 1]
#     plt.plot(x0, (-b0 - b1 * x0) / b2, color='red')
#     plt.title(name)
#     return risk_train