class AutoSklearnRegression():
    def __init__(self,
                 appID,
                 models=None,
                 evaluation_parameters=None,
                 dimensionality_reduction=None
                 ):
        if(models ==  None):
            models=['LinearRegression', 'LinearSVR', 'GradientBoostingRegressor']
        if(evaluation_parameters == None):
            evaluation_parameters=['ExplainedVariance','MAE', 'MSE', 'R2Score']
        if(dimensionality_reduction == None):
            dimensionality_reduction =['ExtraTreesClassifier']
        self.models = models
        self.dimensionality_reduction = dimensionality_reduction
        self.evaluation_parameters = evaluation_parameters
        self.models_to_use = [True, True, True]
        self.params_to_use = [True, True, True, True]
        self.set_training_parameters()
        from sklearn.linear_model import LinearRegression
        self.lr_estimator = LinearRegression()
        from sklearn.svm import LinearSVR
        self.svr_estimator = LinearSVR()
        from sklearn.ensemble import GradientBoostingRegressor
        self.gbr_estimator = GradientBoostingRegressor()
        self.appID = str(appID) + '.html'

    def set_training_parameters(self):
        if 'LinearRegression' not in self.models:
            self.models_to_use[0] = False
        if 'LinearSVR' not in self.models:
            self.models_to_use[1] = False
        if 'GradientBoostingRegressor' not in self.models:
            self.models_to_use[2] = False
        if 'ExplainedVariance' not in self.evaluation_parameters:
            self.params_to_use[0] = False
        if 'MAE' not in self.evaluation_parameters:
            self.params_to_use[1] = False
        if 'MSE' not in self.evaluation_parameters:
            self.params_to_use[2] = False
        if 'R2Score' not in self.evaluation_parameters:
            self.params_to_use[3] = False

    def train(self, data, test_size, response_col_name):
        self.test_size = test_size
        self.data = data
        self.original_data = data
        self.response = response_col_name
        
        self.data=self.data.dropna(axis=1, how='all')

        # Imputing nan
        import numpy as np
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.missing_value_columns = self.data.columns[self.data.isnull().any()]
        for col in self.missing_value_columns:
            if (len(self.data[col].value_counts()) < 5):
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            else:
                self.data[col].fillna(self.data[col].mean(), inplace=True)

        # Encoding data
        from automl.sklearn.preprocessing.DummyEncode import DummyEncode
        self.data = DummyEncode(self.data).encode()

        # Sepearting target and response data
        self.Y = self.data[self.response]
        self.X = self.data.drop(self.response, 1)

        # Dimensionality Reduction
        self.X = self.select_dimensions()

        # Spliting into train and test data set
        from automl.sklearn.preprocessing.Split import Split
        self.x_train, self.x_test, self.y_train, self.y_test = Split().train_test_split(self.X, self.Y, test_size=test_size, random_state=0)

        # Training all selected models with train data
        self.headers = ['Evaluation Parameters']
        generated_models = []
        if self.models_to_use[0]:
            self.lr_model = self.lr_estimator.fit(self.x_train, self.y_train)
            self.headers.append('LinearRegression')
            generated_models.append(self.lr_model)
        if self.models_to_use[1]:
            self.svr_model = self.svr_estimator.fit(self.x_train, self.y_train)
            self.headers.append('LinearSVR')
            generated_models.append(self.svr_model)
        if self.models_to_use[2]:
            self.gbr_model = self.gbr_estimator.fit(self.x_train, self.y_train)
            self.headers.append('GradientBoostingRegressor')
            generated_models.append(self.gbr_model)

        # Predicting on test data with all selected models
        self.predict_all()

        # Print data
        print_data_str = self.print_data()

        # Generating summary of prediction
        print_summary_str = self.summary()

        # Selecting best model on the basis of R2_score
        self.best_model()

        # Returning all selected trained models
        # return generated_models

        # printing output in a html file and storing it in string variable
        with open(self.appID, 'w') as f:
            print(print_data_str , print_summary_str , self.best_model_str, file=f)
        with open(self.appID, 'r') as myfile:
            str_output = myfile.read()

        return str_output

    def select_dimensions(self):
        if self.dimensionality_reduction[0] == 'LinearSVC':
            from automl.sklearn.preprocessing.LinearSVC import LinearSVC
            self.reducer = LinearSVC(self.X,self.Y)
            self.method_used = 'LinearSVC'
        if self.dimensionality_reduction[0] == 'ExtraTreesClassifier':
            from automl.sklearn.preprocessing.ExtraTreesClassifier import ExtraTreesClassifier
            self.reducer = ExtraTreesClassifier(self.X,self.Y)
            self.method_used = 'ExtraTreesClassifier'
        if self.dimensionality_reduction[0] == 'LogisticRegression':
            from automl.sklearn.preprocessing.LogisticRegression import LogisticRegression
            self.reducer = LogisticRegression(self.X,self.Y)
            self.method_used = 'LogisticRegression'
        if self.dimensionality_reduction[0] == 'LassoRegression':
            from automl.sklearn.preprocessing.LassoRegression import LassoRegression
            self.reducer = LassoRegression(self.X,self.Y)
            self.method_used = 'LassoRegression'
        return self.reducer.selectFeatures()

    def predict_all(self):
        self.y_pred_all = []
        if self.models_to_use[0]:
            self.y_predict__lr = self.lr_estimator.predict(self.x_test)
            self.y_pred_all.append(self.y_predict__lr)
        if self.models_to_use[1]:
            self.y_predict__svr = self.svr_estimator.predict(self.x_test)
            self.y_pred_all.append(self.y_predict__svr)
        if self.models_to_use[2]:
            self.y_predict__gbr = self.gbr_estimator.predict(self.x_test)
            self.y_pred_all.append(self.y_predict__gbr)
        # return self.y_predict__lr, self.y_predict__svr, self.y_predict__gbr

    def predict(self, data):
        # Imputing nan
        missing_value_columns = data.columns[data.isnull().any()]
        for col in missing_value_columns:
            if (len(data[col].value_counts()) < 10):
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].mean(), inplace=True)

        # Encoding data
        from automl.sklearn.preprocessing.DummyEncode import DummyEncode
        data = DummyEncode(data).encode()

        if self.response in data.columns:
            data_y = data[self.response]
            data_x = data.drop(self.response, 1)
        else:
            data_y = None
            data_x = data
        if self.best_fit_model == self.lr_estimator:
            self.predcition = self.lr_estimator.predict(data_x)
        elif self.best_fit_model == self.svr_estimator:
            self.prediction = self.svr_estimator.predict(data_x)
        else:
            self.prediction = self.gbr_estimator.predict(data_x)
        return self.predcition

    def score(self, x_train, y_train):
        return self.lr_estimator.score(x_train, y_train), self.svr_estimator.score(x_train, y_train), self.gbr_estimator.score(x_train, y_train)

    def summary(self):
        from tabulate import tabulate
        from sklearn.metrics import explained_variance_score
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score
        evaluation_table = []
        if self.params_to_use[0]:
            evaluation_table.append(['Explained Variance Score'])
        if self.params_to_use[1]:
            evaluation_table.append(['Mean Absolute Error'])
        if self.params_to_use[2]:
            evaluation_table.append(['Mean Squared Error'])
        if self.params_to_use[3]:
            evaluation_table.append(['R2 Score'])
        for y_pred in self.y_pred_all:
            i = 0
            if self.params_to_use[0]:
                evaluation_table[i].append(explained_variance_score(self.y_test, y_pred))
                i = i + 1
            if self.params_to_use[1]:
                evaluation_table[i].append(mean_absolute_error(self.y_test, y_pred))
                i = i + 1
            if self.params_to_use[2]:
                evaluation_table[i].append(mean_squared_error(self.y_test, y_pred))
                i = i + 1
            if self.params_to_use[3]:
                evaluation_table[i].append(r2_score(self.y_test, y_pred))
        summary_str = '<p><b>Accuracy Metric:</b></p><div style="overflow-x:auto;">'\
                      + tabulate(evaluation_table, headers=self.headers ,tablefmt="html")+ '</div>'
        return summary_str

    def print_data(self):
        tb_data = self.original_data.head(n=5)
        tb_train = self.x_train.head(n=5)
        tb_test = self.x_test.head(n=5)
        tb_columns1 = list(self.data.columns)
        tb_columns2 = list(self.x_train.columns)
        style_html = '<style>table {border-collapse: collapse;width: 80%;}th, td {padding: 8px;text-align: left;border: 1px solid #ddd; font-size: 12px;}tr:hover {background-color:#f5f5f5;}th {background-color: #ec1a3d;color: white;}</style>'
        from tabulate import tabulate
        info_tables = '<div><p><b>Data Dimensions: </b>' + str(self.data.shape[0]) \
                      + ' Rows and ' + str(self.data.shape[1]) + ' Features</p><p><b>Prediction Variable: </b>' \
                      + self.response + '</p>' + '<p><b>Available Features: </b>' + str(tb_columns1) + '</p>' \
                      + '<p><b>Columns where missing values were found and replaced: </b>' \
                      + str(self.missing_value_columns) + '</p>' +'<p><b>Method Used for Dimensionality Reduction:</b> '+ self.method_used +'</p><p><b>Selected Best Features: </b>' \
                      + str(tb_columns2) + '</p>' + '<p><b>Complete Dataset: </b>' + str(self.data.shape[0]) \
                      + ' Rows and ' + str(self.data.shape[1]) + ' Features</p>' + '<p><b>Target Feature: </b>' \
                      + self.response + '</p>' + '<div style="overflow-x:auto;">' \
                      + tabulate(tb_data, headers=tb_columns1, tablefmt="html") + '</div>' \
                      + '<p><b>Splitting:</b> [############] 100%</p>' + '<p><b>Training Dataset: </b>' \
                      + str(self.x_train.shape[0]) + ' Rows and ' + str(self.x_train.shape[1]) + ' Features</p>' \
                      + '<div style="overflow-x:auto;">' + tabulate(tb_train, headers=tb_columns2, tablefmt="html") \
                      + '</div>' + '<p><b>Testing Dataset: </b>' + str(self.x_test.shape[0]) + ' Rows and ' \
                      + str(self.x_test.shape[1]) + ' Features</p>' + '<div style="overflow-x:auto;">' \
                      + tabulate(tb_test, headers=tb_columns2, tablefmt="html") + '</div>' \
                      + '<p><b>Training Models:</b> [#############################] 100%</p>'
        return style_html + info_tables

    def best_model(self):
        acc_lr, acc_svr, acc_gbr = -10000, -10000, -10000
        from sklearn.metrics import r2_score
        if self.models_to_use[0]:
            acc_lr = r2_score(self.y_test, self.y_predict__lr)
        if self.models_to_use[1]:
            acc_svr = r2_score(self.y_test, self.y_predict__svr)
        if self.models_to_use[2]:
            acc_gbr = r2_score(self.y_test, self.y_predict__gbr)
        best_acc = max(acc_lr, acc_svr, acc_gbr)
        if best_acc == acc_lr:
            model_name = 'Linear Regression'
            self.best_fit_model = self.lr_estimator
        elif best_acc == acc_svr:
            model_name = 'Support Vector Regression'
            self.best_fit_model = self.svr_estimator
        else:
            model_name = 'Gradient Boosting Regression'
            self.best_fit_model = self.gbr_estimator
        self.best_model_str = '<p><b>Evaluating Best Model:</b> [##########] Done</p><p><b>Best Model:</b> ' \
                              + model_name + '</p><p><b>R2_Score:</b> ' + str(best_acc) + '</p></div>'
        return self.best_fit_model