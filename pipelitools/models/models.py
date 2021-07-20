import numpy as np
import pandas as pd
import random
import pickle
import os

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from pipelitools.models import metrics as m


def test_models():
    """ """
    print('test_models: ok')


class Model:
    """Runs a model, plots confusion matrix, calculates the metrics and outputs the reports in a folder.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features used in training.
    y_train : pd.Series
        Labels for training (1D vector).
    X_test : pd.DataFrame
        Features used in testing.
    y_test : pd.Series
        Labels for testing (1D vector).

    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def checkmodel(self,
                   name,
                   model,
                   steps=[],
                   parameters={},
                   average='binary',
                   multiclass=False,
                   metric='accuracy',
                   randomized_search=False,
                   nfolds=5,
                   n_jobs=None,
                   save_pickle=True,
                   verbose=0
                   ):
        """ Calculates the model based on the pipeline and hyperparameter grid.
        Then, evaluates metrics (f1-score, accuracy, precision, recall) and plots a confusion matrix.
        Can save the final fitted model with pickle to load later.

        Parameters
        ----------
        name : str
            Name of the model.
        model : abc.ABCMeta
            Machine learning model.
        steps : list, optional (default = [])
            Steps of the preprocessing pipeline.
        parameters : dict, optional (default = {})
            Parameters of the model.
        average : str, optional (default = 'binary')
            This parameter is required for multiclass/multilabel targets. If None, the scores for each class are
            returned. Otherwise, this determines the type of averaging performed on the data
        multiclass : bool, optional (default = False)
            True if the classification is multiclass.
        metric : str, optional (default = 'accuracy')
            Metric which should be used to select the best model.
        randomized_search : bool, optional (default = False)
            True if randomized search.
        nfolds : int, optional (default = 5)
            Number of folds in CV.
        n_jobs : int, optional (default = None)
            The number of parallel jobs to run.
        save_pickle : bool, optional (default=True)
            Save the best fitted model with pickle.
            To load do:
            loaded_model = pickle.load(open('./pickle_models/model.sav', 'rb'))
        verbose : int, optional (default = 0)
            Verbose CV.

        Returns
        -------
        cv : sklearn.model_selection._search.GridSearchCV
            The fitted model.
        y_pred : np.ndarray
            predicted values.
        Figures are saved in a separate folder.
        """
        assert ' ' not in name, "Parameter 'name' must be specified without space inside."
        assert isinstance(self.y_train, pd.Series), "y_train must be of type pd.Series."
        assert isinstance(self.y_test, pd.Series), "y_test must be of type pd.Series."

        if len(parameters) != 0:
            random_parameter = random.choice(list(parameters.keys()))
            assert '__' in random_parameter and name in random_parameter, \
                f"Parameters should be presented in a dictionary in the following way: \n\
            '{name}__parameter': [parameter_value]"

        steps_model = steps[:]

        # Create the pipeline
        if multiclass:
            from imblearn.pipeline import Pipeline
        else:
            from sklearn.pipeline import Pipeline

        steps_model.append((name, model))
        pipeline = Pipeline(steps_model)

        if multiclass:
            cv_metric = metric + '_'+average
        else:
            cv_metric = metric

        if randomized_search:
            cv = RandomizedSearchCV(estimator=pipeline,
                                    param_distributions=parameters,
                                    cv=nfolds,
                                    # refit=cv_metric',
                                    scoring=cv_metric,
                                    # n_iter=10,
                                    verbose=verbose,
                                    n_jobs=n_jobs,
                                    random_state=42)

        else:
            cv = GridSearchCV(estimator=pipeline,
                              param_grid=parameters,
                              cv=nfolds,
                              # refit=cv_metric',
                              scoring=cv_metric,
                              verbose=verbose,
                              n_jobs=n_jobs)

        # Fit to the training set
        cv.fit(self.X_train, self.y_train)

        # Mean cross-validated score of the best_estimator
        print(f"Mean cross-validated score of the best_estimator: {round(cv.best_score_, 4)}")

        # Parameter setting that gave the best results on the validation data
        if len(parameters) != 0:
            df_tuned = pd.DataFrame(cv.best_params_, index=[0]).transpose().reset_index().rename(
                columns={'index': 'Parameter', 0: 'Tuned value'})
            df_tuned['Parameter'] = df_tuned.Parameter.str.partition('__').iloc[:, -1]
            print(df_tuned, '\n')

        # Predict the labels of the test set
        y_pred = cv.predict(self.X_test)

        # METRICS
        m.metrics_report(cv, name, self.X_test, self.y_test, self.y_train, data='validation')

        # SAVE MODEL USING PICKLE
        if save_pickle:
            if os.path.exists("./temp_pickle_models/") is False:
                os.mkdir("./temp_pickle_models/")

            pickle.dump(cv, open(f"./temp_pickle_models/{name}.sav", 'wb'))

        return cv, y_pred

    def evaluate(self, model, name, X_test, y_test, y_train):
        m.metrics_report(model, name, X_test, y_test, y_train, data='test')


if __name__ == '__main__':
    test_models()
