from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 225)

oldie = pd.read_csv('IMDB files/imputed_movies_people_data.csv', header=0, low_memory=False, index_col=0)
# newie = pd.read_csv('New IMDB files/imputed_movies_people_data.csv', header=0, low_memory=False, index_col=0)
#
# newie = newie[~newie.isin(oldie)].dropna()
#
# newie = newie.drop(['x0_nm0372366', 'x0_nm0790144', 'x0_nm0404806', 'x0_nm1698571', 'x0_nm0770650', 'x0_nm0003911',
#                     'x0_nm0023062', 'x0_nm0241489', 'x0_nm0854185', 'x0_nm0905154', 'x0_nm0859211',
#                     'x0_nm1206265', 'x0_nm3571592', 'x0_nm2222628', 'x0_nm0661406', 'x0_nm0297209'], axis=1)
#
# main_list = list(set(oldie.columns) - set(newie.columns))
# for column in main_list:
#     newie[column] = 0
#
# newie.to_csv('New IMDB files/different_data.csv')
newie = pd.read_csv('New IMDB files/different_data.csv', header=0, low_memory=False, index_col=0)
loaded_model = pickle.load(open("XGBoost Model", 'rb'))

cols_when_model_builds = loaded_model.get_booster().feature_names
y_test = newie.averageRating
X_test = newie[cols_when_model_builds]

s = (X_test.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:" + str(object_cols))
X_test = X_test.drop(object_cols, axis=1)

preds = loaded_model.predict(X_test)
print(r2_score(y_test, preds))

mae = mean_absolute_error(y_test, preds)
print(str(mae))
y_test.sort_values(ascending=True)

plt.plot(preds, y_test, 'o')
m, b = np.polyfit(preds, y_test, 1)
plt.plot(preds, m * preds + b)
plt.title('Predictions Vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.xlim([1, 10])
plt.ylim([1, 10])

plt.show()
