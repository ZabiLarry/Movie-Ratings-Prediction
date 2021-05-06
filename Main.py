import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import winsound
from sklearn.pipeline import Pipeline
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 225)
log = open("log.txt", "a")
MIN_MOVIES = 10
MIN_PEOPLE = 10


def genres_region_one_hot_encoder(df):
    from sklearn.preprocessing import MultiLabelBinarizer
    counts = df.region.value_counts()
    repl = counts[counts <= MIN_MOVIES].index
    regions_encoded = pd.get_dummies(df.region.replace(repl, 'uncommonRegion'), columns=counts.index)
    mlb = MultiLabelBinarizer()
    a = mlb.fit_transform(df['genres'].str.split(','))
    genres_encoded = pd.DataFrame(a, columns=mlb.classes_, index=df.index)
    genres_encoded.drop(['Adult', 'Reality-TV', 'Short'
                         # , 'Talk-Show'
                         ], axis=1, inplace=True)  # talk show might not be there
    # print(genres_encoded.apply(pd.Series.value_counts, axis=0)) #  for visuals
    genres_encoded = pd.merge(df, genres_encoded, on="titleId", how="inner")
    genres_encoded = genres_encoded.drop('genres', axis=1)  # dropped genres
    encoded = pd.merge(genres_encoded, regions_encoded, on="titleId", how="inner")
    encoded = encoded.drop('region', axis=1)
    from sklearn.utils import shuffle
    encoded = shuffle(encoded, random_state=0)
    print("regions and genres encoded")
    encoded.to_csv('IMDB files/encoded_movies_data.csv')
    return encoded


def score_dataset_RFR(X_train, X_valid, y_train, y_valid):
    from sklearn.ensemble import RandomForestRegressor
    print("Random Forest Regressor started")
    n_estimators = 1000
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    print("MAE RFR: " + str(mean_absolute_error(y_valid, preds)))
    print("n_estimators=" + str(n_estimators))
    log.write("\n MAE RFR: " + str(mean_absolute_error(y_valid, preds)) +
              "\n n_estimators=" + str(n_estimators))
    check_importance(model, X_train)


def check_importance(model, X):
    importances = list(model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(X.columns, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical', color='r', edgecolor='k', linewidth=1.2)
    # Tick labels for x axis
    plt.xticks(x_values, X.columns, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()


def score_dataset_XGB(X_train, X_valid, y_train, y_valid):
    from xgboost import XGBRegressor
    print("XGBRegressor started")
    n_estimators = 1000
    learning_rate = 0.05
    n_jobs = 4
    early_stopping_rounds = 5
    my_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=n_jobs)
    my_model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_set=[(X_valid, y_valid)],
                 verbose=False)
    preds = my_model.predict(X_valid)
    print("MAE XGB: " + str(mean_absolute_error(y_valid, preds)))
    print(
        "n_estimators=" + str(n_estimators) + ", learning_rate=" + str(learning_rate) + ", n_jobs=" + str(
            n_jobs) + ", early_stopping_rounds=" + str(early_stopping_rounds))
    log.write("\n MAE XGB: " + str(mean_absolute_error(y_valid, preds)) +
              "\n n_estimators=" + str(n_estimators) + ", learning_rate=" + str(learning_rate) + ", n_jobs=" + str(
        n_jobs) + ", early_stopping_rounds=" + str(early_stopping_rounds))


def score_dataset_KNN(X_train, X_valid, y_train, y_valid):
    from sklearn.neighbors import KNeighborsRegressor
    print("KNNeighbor started")
    n_neighbors = 5
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    preds = knn_model.predict(X_valid)
    print("MAE KNN: " + str(mean_absolute_error(y_valid, preds)))
    print("n_neighbors=" + str(n_neighbors))
    log.write("\n MAE KNN: " + str(mean_absolute_error(y_valid, preds)) + "\n n_neighbors=" + str(n_neighbors))


def score_dataset_DTR(X_train, X_valid, y_train, y_valid):
    from sklearn.tree import DecisionTreeRegressor
    print("Decision Tree Regressor started")
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    print("MAE DTR: " + str(mean_absolute_error(y_valid, preds)))
    log.write("\n MAE DTR: " + str(mean_absolute_error(y_valid, preds)))


def score_dataset_RR(X_train, X_valid, y_train, y_valid):
    from sklearn import linear_model
    print("Ridge Regression started")
    model = linear_model.RidgeCV()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    print("MAE RR: " + str(mean_absolute_error(y_valid, preds)))
    log.write("\n MAE RR: " + str(mean_absolute_error(y_valid, preds)))


def score_dataset_LR(X_train, X_valid, y_train, y_valid):
    from sklearn import linear_model
    print("Lasso Regression started")
    model = linear_model.LassoCV()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    print("MAE LR: " + str(mean_absolute_error(y_valid, preds)))
    log.write("\n MAE LR: " + str(mean_absolute_error(y_valid, preds)))


def score_dataset_ANN(X_train, X_valid, y_train, y_valid):
    import tensorflow as tf
    from tensorflow import keras
    print("Artificial Neural Network started")
    epochs = 100
    patience = 5
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),  # input layer
        keras.layers.Dense(64, activation='relu'),  # hidden layer (1)
        keras.layers.Dense(64, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(1)  # output layer
    ])
    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, callbacks=[callback])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    preds = model.predict(X_valid)
    print("MAE ANN: " + str(mean_absolute_error(y_valid, preds)))
    print(" epochs=" + str(epochs))
    log.write("\n MAE ANN: " + str(mean_absolute_error(y_valid, preds)) +
              "\n epochs=" + str(epochs))


def get_scores(df):
    from sklearn.model_selection import train_test_split
    SHUFFLE = False
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    print("Categorical variables:" + str(object_cols))
    df = df.drop(object_cols, axis=1)
    # label_encoder = LabelEncoder()
    # for col in object_cols:
    #     df[col] = label_encoder.fit_transform(df[col])

    print("MIN_MOVIES = " + str(MIN_MOVIES) + ",  MIN_PEOPLE = " + str(MIN_PEOPLE) + " shuffle=" + str(SHUFFLE))
    log.write("\nMIN_MOVIES = " + str(MIN_MOVIES) + ",  MIN_PEOPLE = " + str(MIN_PEOPLE) + " shuffle=" + str(SHUFFLE))
    y = df.averageRating
    X = df.drop(['averageRating'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0,
                                                          shuffle=SHUFFLE)

    score_dataset_RFR(X_train, X_valid, y_train, y_valid)
    # score_dataset_XGB(X_train, X_valid, y_train, y_valid)
    # score_dataset_KNN(X_train, X_valid, y_train, y_valid)
    # score_dataset_DTR(X_train, X_valid, y_train, y_valid)
    # score_dataset_LR(X_train, X_valid, y_train, y_valid)
    # score_dataset_RR(X_train, X_valid, y_train, y_valid)
    # score_dataset_ANN(X_train, X_valid, y_train, y_valid)


def add_people(movies_data):
    print("started adding people")
    principals_data = pd.read_csv('IMDB files/clean_principals_data.csv', header=0, low_memory=False, index_col=0)

    # principals_data = principals_data.loc[principals_data['category'] == 'director']
    count = principals_data.nconst.value_counts()
    count = count[count.values >= MIN_PEOPLE]
    principals_data = principals_data[
        principals_data.nconst.isin(count.index)]
    from sklearn.preprocessing import OneHotEncoder
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    principals_encoded = pd.DataFrame(OH_encoder.fit_transform(principals_data[['nconst']]),
                                      index=principals_data.index)
    OH_encoder.get_feature_names()
    principals_encoded.columns = OH_encoder.get_feature_names()
    principals_encoded = principals_encoded.reset_index().groupby('titleId').max()
    # principals_encoded.drop_duplicates(subset=principals_data.columns[[0]], inplace=True)
    movies_people_data = pd.merge(movies_data, principals_encoded, on="titleId", how="left")
    imputer = SimpleImputer(missing_values=np.NaN, strategy='constant', fill_value=0.0)
    imputed_movies_people_data = pd.DataFrame(imputer.fit_transform(movies_people_data))
    imputed_movies_people_data.columns = movies_people_data.columns
    imputed_movies_people_data.index = movies_people_data.index
    imputed_movies_people_data.to_csv('IMDB files/imputed_movies_people_data.csv')
    print("people added")
    return imputed_movies_people_data


def reset():
    movies_data = pd.read_csv('IMDB files/clean_movies_data.csv', header=0, low_memory=False, index_col=1)
    movies_data = movies_data.drop(movies_data.columns[[0]], axis=1)  # drop useless column (6 if akas)
    #  missing_runtime = movies_data.loc[movies_data['runtimeMinutes'].isnull() | movies_data['genres'].isnull()]
    #  print(missing_runtime.shape)
    movies_data.dropna(how='any', subset=['runtimeMinutes', 'genres'], inplace=True)  # must replace \N first
    # genres_vals = movies_data.genres.value_counts()
    # print(genres_vals)

    # pg_movies_data = movies_data.loc[movies_data['isAdult'] == 1]
    # sorted = pg_movies_data.sort_values(by='averageRating', ascending=False)
    # print(sorted.head(20))

    add_people(genres_region_one_hot_encoder(movies_data))


# reset()
# print(movies_data.isnull().sum())
get_scores(pd.read_csv('IMDB files/encoded_movies_data.csv', header=0, low_memory=False, index_col=0))

# get_scores(pd.read_csv('IMDB files/imputed_movies_people_data.csv', header=0, low_memory=False, index_col=0))

winsound.Beep(1000, 800)
winsound.Beep(900, 500)
