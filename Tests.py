import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)

df = pd.read_csv('IMDB files/imputed_movies_people_data.csv', header=0, low_memory=False, index_col=0)

print(df.columns)
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:" + str(object_cols))
df = df.drop(object_cols, axis=1)

y = df.averageRating
X = df.drop(['averageRating'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


def score_dataset_LR(X_train, X_valid, y_train, y_valid):
    from sklearn import linear_model
    print("Linear Regression started")
    model = linear_model.LassoCV()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    print("MAE LR: " + str(mean_absolute_error(y_valid, preds)))


# score_dataset_LR(X_train, X_valid, y_train, y_valid)


def cross_validation():
    from sklearn.model_selection import cross_val_score
    df = pd.read_csv('IMDB files/encoded_movies_data.csv', header=0, low_memory=False, index_col=0)
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    print("Categorical variables:" + str(object_cols))
    df = df.drop(object_cols, axis=1)
    y = df.averageRating
    X = df.drop(['averageRating'], axis=1)
    pipeline = Pipeline(steps=[('model', RandomForestRegressor(n_estimators=100, random_state=0))])
    scores = -1 * cross_val_score(pipeline, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    print("MAE scores:\n", scores)


def count_genres():
    movies_data = pd.read_csv('IMDB files/clean_movies_data.csv', header=0, low_memory=False, index_col=1)
    counts = movies_data.region.value_counts()
    counts = counts[counts.values >= 10]
    print(counts.shape)


# count_genres()


def count_people():
    principals_data = pd.read_csv('IMDB files/clean_principals_data.csv', header=0, low_memory=False, index_col=0)

    principals_data = principals_data.loc[principals_data['category'] == 'actor']
    count = principals_data.nconst.value_counts()
    count = count[count.values >= 10]
    principals_data = principals_data[
        principals_data.nconst.isin(count.index)]
    print(count)


# count_people()

# from sklearn.preprocessing import OneHotEncoder
# OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# principals_encoded = pd.DataFrame(OH_encoder.fit_transform(principals_data[['nconst']]), index=principals_data.index)
# OH_encoder.get_feature_names()
# principals_encoded.columns = OH_encoder.get_feature_names()
# print(principals_encoded.head())


#  year_data = year_data.sort_values(year_data.columns[0]).to_csv('IMDB files/startYear.csv')
def plot_year_movies():
    movies_data = pd.read_csv('IMDB files/movies_data_vis.csv', header=0, low_memory=False, index_col=1)
    year_data = pd.read_csv('IMDB files/startYear.csv', header=0, low_memory=False, index_col=1)
    movies_data = movies_data.dropna(how='any', subset=['startYear'])
    year_data = year_data.drop(year_data.columns[[0]], axis=1)
    plt.plot(year_data)
    plt.xlabel("Year")
    plt.ylabel("Movie Amount")
    plt.show()
