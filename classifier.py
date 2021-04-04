import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from catboost import CatBoostClassifier, Pool
from configs import config_vars


class ModelExplained:
    def __init__(self, iteration_range=[50, 100, 500]):
        self.feature_list = config_vars['feature_list']
        self.data = config_vars['data']
        self.iteration_range = iteration_range
        self.model_features = None
        self.category_features = None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.trained_model = None

    @staticmethod
    def extract_domain(x) -> str:
        """
        Extract domain name
        """
        return x[(x.index('@') + 1):x.index('.')]

    # add imputation feature
    @staticmethod
    def data_imputation(dataframe, features: dict):
        """
        Fill missing values
        """
        for key, value in features.items():
            if value['model']:
                if value['type'] == 'numeric':
                    param = dataframe[key].median()
                else:
                    param = dataframe[key].mode()[0]
                dataframe[key] = dataframe[key].fillna(param)
        return dataframe

    def feature_extraction(self) -> bool:
        """
        Config model and categories features on a class level
        """
        self.model_features = [key for key, value in self.feature_list.items() if value['model']]
        self.category_features = [key for key, value in self.feature_list.items()
                                  if value['type'] == 'string' and key != 'y' and value['model']]
        return True

    def data_preps(self) -> bool:
        """
        Split data into train and test datasets and save on a class level
        """
        self.data['y'] = [0 if item == 'approved' else 1 for item in self.data['status'].to_list()]
        self.data['email_anoni'] = [item for item in self.data['email_anoni'].to_list()]
        X = self.data
        y = self.data['y']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.X_train = self.data_imputation(self.X_train, self.feature_list)[self.model_features]
        self.X_test = self.data_imputation(self.X_test, self.feature_list)
        return True

    def param_model_training(self, learning_rate: float, depth: int, trees: int) -> tuple:
        """
        Training a model for a given hyper params
        Returns: model, model predictions and probs
        """
        X = self.X_train
        y = self.y_train
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

        clf = CatBoostClassifier(
            iterations=trees,
            learning_rate=learning_rate,
            depth=depth,
        )

        clf.fit(X_train, y_train,
                cat_features=self.category_features,
                eval_set=(X_val, y_val),
                verbose=False)

        return clf, clf.predict(data=X_val), clf.predict_proba(data=X_val), y_val

    def hyper_params(self) -> dict:
        """
        Grid search for hyper params
        Returns: dictionary with evaluation metrics
        """
        hyper_params = {
            'learning_rate': [],
            'depth': [],
            'trees': [],
            'log_loss': [],
            'prauc': [],
            'roc_auc': []
        }
        lr_range = np.arange(0.16, 0.19, 0.01)
        depth_range = range(7, 9)
        tree_range = self.iteration_range
        for lr in lr_range:
            for d in depth_range:
                for t in tree_range:
                    clf, predicted_prob, predicted_class, y_true = self.param_model_training(lr, d, t)
                    prauc = round(metrics.average_precision_score(y_true, predicted_prob, pos_label=1), 5)
                    roc_auc = metrics.roc_auc_score(y_true, predicted_prob)
                    logloss = round(metrics.log_loss(y_true, predicted_prob), 5)
                    hyper_params['learning_rate'].append(lr)
                    hyper_params['depth'].append(d)
                    hyper_params['trees'].append(t)
                    hyper_params['log_loss'].append(logloss)
                    hyper_params['prauc'].append(prauc)
                    hyper_params['roc_auc'].append(roc_auc)
        return hyper_params

    def main_model_training(self):
        """
        Training a model with hyper params that produce the highest PRAUC score
        Returns: trained model
        """
        params_data = self.hyper_params()
        df_params = pd.DataFrame(params_data)
        df_params.to_csv('data/plots/hyper_params.csv')
        res = df_params[df_params.prauc == max(df_params.prauc)].iloc[0]
        learning_rate, depth, trees = res[0], res[1], res[2]

        clf = CatBoostClassifier(
            iterations=trees,
            learning_rate=learning_rate,
            depth=depth,
        )

        trained_model = clf.fit(
            self.X_train,
            self.y_train,
            cat_features=self.category_features,
            verbose=False
        )
        self.trained_model = trained_model
        return True

    @staticmethod
    def get_approval_rate(th, scores: list) -> float:
        """
        Returns: approval rate for a given threshold and model scores
        """
        th_class = [1 if item[1] > th else 0 for item in scores]
        results = {i: th_class.count(i) for i in set(th_class)}
        keys = [item for item in results.keys()]
        if len(keys) == 2:
            return results[0]*100.0/len(th_class)
        else:
            return 100 if 0 in keys else 0

    def find_threshold(self, th, required_approval_rate: float = 0.9) -> bool:
        """
        Finding the threshold that produce the required approval rate.
        Save plot data in relevant folder
        """
        clf = self.trained_model
        if clf is not None:
            model_probs = self.trained_model.predict_proba(self.X_test[self.model_features])
            approval_rate = self.get_approval_rate(th, model_probs.tolist())
            th_l, app_rate_l = [th], [approval_rate]
            while abs(approval_rate - required_approval_rate) > 0.15 and th < 1:
                th += 0.001
                approval_rate = self.get_approval_rate(th, model_probs)
                th_l.append(th)
                app_rate_l.append(approval_rate)
            df = pd.DataFrame({'approval_rate': app_rate_l, 'threshold': th_l})
            df.to_csv('data/plots/threshold.csv')
            return True
        else:
            return False

    def produce_plot_datasets(self):
        """
        Save PRAUC data and test dataset with model predictions in relevant folder.
        """
        try:
            df_main = pd.DataFrame(self.X_test)
            predicted_results = self.trained_model.predict_proba(data=self.X_test[self.model_features]).tolist()
            prob_0 = [item[0] for item in predicted_results]
            prob_1 = [item[1] for item in predicted_results]
            df_main['model_score_0'] = prob_0
            df_main['model_score_1'] = prob_1
            df_main.to_csv('data/plots/dataset_test.csv')
            precision, recall, _ = metrics.precision_recall_curve(df_main['y'], prob_1)
            df_prauc = pd.DataFrame({'precision': precision, 'recall': recall})
            df_prauc.to_csv('data/plots/dataset_prauc.csv')
        except Exception as e:
            print(e)

    def main(self):
        self.feature_extraction()
        self.data_preps()
        self.main_model_training()
        self.find_threshold(th=0.2)
        self.produce_plot_datasets()

