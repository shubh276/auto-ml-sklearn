class AutoSklearnClassification():
    
    def __init__(self,
                 appID,
                 models=None,
                 dimensionality_reduction=None
                 ):
        if (dimensionality_reduction == None):
            dimensionality_reduction = ['ExtraTreesClassifier']
        if(models == None):
            models=['LogisticRegression', 'SVC', 'GradientBoostingClassifier']
        self.dimensionality_reduction = dimensionality_reduction
        self.models = models
        self.models_to_use = [True, True, True]
        self.appID = str(appID) + '.html'
        self.set_training_parameters()
        from sklearn.linear_model import LogisticRegression
        self.lr_estimator = LogisticRegression()
        from sklearn import svm
        self.svc_estimator = svm.SVC()
        from sklearn.ensemble import GradientBoostingClassifier
        self.xgbc_estimator = GradientBoostingClassifier()
    
    def set_training_parameters(self):
        if 'LogisticRegression' not in self.models:
            self.models_to_use[0] = False
        if 'SVC' not in self.models:
            self.models_to_use[1] = False
        if 'GradientBoostingClassifier' not in self.models:
            self.models_to_use[2] = False

    def train(self, data, test_size, response_col_name):
        self.test_size = test_size
        self.data = data
        self.original_data = data
        self.response = response_col_name
        self.data=self.data.dropna(axis=1, how='all')
        
        # Imputing nan/missing values
        import numpy as np
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.missing_value_columns = self.data.columns[self.data.isnull().any()]
        for col in self.missing_value_columns:
            if (len(self.data[col].value_counts()) < 10):
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
        generated_models = []
        if self.models_to_use[0]:
            self.lr_model = self.lr_estimator.fit(self.x_train, self.y_train)
            generated_models.append(self.lr_model)
        if self.models_to_use[1]:
            self.svc_model = self.svc_estimator.fit(self.x_train, self.y_train)
            generated_models.append(self.svc_model)
        if self.models_to_use[2]:
            self.xgbc_model = self.xgbc_estimator.fit(self.x_train, self.y_train)
            generated_models.append(self.xgbc_model)

        # Predicting on test data with all selected models
        self.predict_all()
        
        # Print data
        print_data_str = self.print_data()
        
        # Generating summary of prediction
        print_summary_str = self.summary()
        
        # Selecting best model on the basis of Accuracy
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
            self.y_predict__svc = self.svc_estimator.predict(self.x_test)
            self.y_pred_all.append(self.y_predict__svc)
        if self.models_to_use[2]:
            self.y_predict__xbgc = self.xgbc_estimator.predict(self.x_test)
            self.y_pred_all.append(self.y_predict__xbgc)
        # return self.y_predict__lr, self.y_predict__svc, self.y_predict__xbgc

    def predict(self, data):

        # Imputing nan
        import numpy as np
        data = data.replace([np.inf, -np.inf], np.nan)
        missing_value_columns = data.columns[data.isnull().any()]
        for col in missing_value_columns:
            if (len(data[col].value_counts()) < 10):
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].mean(), inplace=True)

        # Encoding data
        from automl.sklearn.preprocessing.DummyEncode import DummyEncode
        data = DummyEncode(data).encode()

        # Seperating target column
        if self.response in data.columns:
            data_y = data[self.response]
            data_x = data.drop(self.response, 1)
        else:
            data_y = None
            data_x = data

        # Predicting using best model
        if self.best_fit_model == self.lr_estimator:
            self.predcition = self.lr_estimator.predict(data_x)
        elif self.best_fit_model == self.svc_estimator:
            self.prediction = self.svc_estimator.predict(data_x)
        else:
            self.prediction = self.xgbc_estimator.predict(data_x)

        return self.predcition

    def confusion_matrix(self, y_true, y_predict):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_predict)

    def summary(self):
        from sklearn.metrics import classification_report
        str0 = '<p><b>Classification Reports:</b></p>'
        str1=""
        str2=""
        str3=""
        if self.models_to_use[0]:
            str1 = self.classifaction_report_table(classification_report(self.y_test, self.y_predict__lr), 'Logistic Regression')
        if self.models_to_use[1]:
            str2 = self.classifaction_report_table(classification_report(self.y_test, self.y_predict__svc), 'Support Vector Classification')
        if self.models_to_use[2]:
            str3 = self.classifaction_report_table(classification_report(self.y_test, self.y_predict__xbgc), 'Gradient Boosting Classification')
        return str0 + str1 + str2 + str3

    def classifaction_report_table(self, report, model_name):
        report_data = []
        lines = report.split('\n')
        del lines[-3]
        for line in lines[2:-1]:
            row = {}
            row_data = line.split('      ')
            i = 1
            if row_data[0] == 'avg / total':
                i = 0
            row['class'] = row_data[i + 0]
            row['precision'] = float(row_data[i + 1])
            row['recall'] = float(row_data[i + 2])
            row['f1_score'] = float(row_data[i + 3])
            row['support'] = float(row_data[i + 4])
            report_data.append(row)
        import pandas as pd
        dataframe = pd.DataFrame.from_dict(report_data)
        from tabulate import tabulate
        report_str = '<p>========================<b>'+ model_name+ 'Report </b>==========================</p><p>'+ tabulate(dataframe, headers=['class', 'precision', 'recall', 'f1_score', 'support'], tablefmt="html")+ '</p>'
        return report_str

    def best_model(self):
        acc_lr, acc_svc, acc_xbgc = -10000, -10000, -10000
        from sklearn.metrics import accuracy_score
        if self.models_to_use[0]:
            acc_lr = accuracy_score(self.y_test, self.y_predict__lr)
        if self.models_to_use[1]:
            acc_svc = accuracy_score(self.y_test, self.y_predict__svc)
        if self.models_to_use[2]:
            acc_xbgc = accuracy_score(self.y_test, self.y_predict__xbgc)
        best_acc = max(acc_lr, acc_svc, acc_xbgc)
        if best_acc == acc_lr:
            model_name = 'Logistic Regression'
            self.best_fit_model = self.lr_estimator
        elif best_acc == acc_svc:
            model_name = 'Support Vector Classification'
            self.best_fit_model = self.svc_estimator
        else:
            model_name = 'Gradient Boosting Classification'
            self.best_fit_model = self.xgbc_estimator
        self.best_model_str = '<p><b>Evaluating Best Model:</b> [##########] Done</p><p><b>Best Model:</b> ' \
                              + model_name + '</p><p><b>Accuracy:</b> ' + str(best_acc) + '</p></div>'
        return self.best_fit_model

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
                      + str(self.missing_value_columns) + '</p>' + '<p><b>Method Used for Dimensionality Reduction:</b> ' + self.method_used + '<p><b>Selected Best Features: </b>' \
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
