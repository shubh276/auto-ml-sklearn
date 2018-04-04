import pandas as pd
appAggregatedData = pd.read_csv("Telstra_Classification_Severity_Prediction.csv")
from AutoSklearnClassification import AutoSklearnClassification
test_size = 0.2
response_col_name = 'severity_type'
# models = [ 'GradientBoostingClassifier', 'LogisticRegression', 'SVC']
models = []
appId = 11112222
dimension_reduction_methods = ['LinearSVC']
# dimension_reduction_methods = []
asc = AutoSklearnClassification(appId, models, dimension_reduction_methods)
train_output = asc.train(appAggregatedData, test_size, response_col_name)
asc.best_model()

import pandas as pd
from AutoSklearnRegression import AutoSklearnRegression
# from previous_versions.AutoSklearnRegression_v9 import AutoSklearnRegression
data = pd.read_csv('HousePrice_Regression_prediction.csv')
test_size = 0.2
response_col_name = 'SalePrice'
models = []
evaluation_parameters = ['ExplainedVariance', 'MSE', 'MAE']
appId = 123241
dimension_reduction_methods = ['LogisticRegression']
asr = AutoSklearnRegression(appId, models,evaluation_parameters, dimension_reduction_methods)
output_str = asr.train(data, test_size, response_col_name)
asr.best_model()