import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import winsound
from sklearn.pipeline import Pipeline
import numpy as np
import os
from sklearn.model_selection import train_test_split

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 225)
MIN_MOVIES = 10
MIN_PEOPLE = 10


def get_scores(df):
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    print("Categorical variables:" + str(object_cols))
    df = df.drop(object_cols, axis=1)
    # label_encoder = LabelEncoder()
    # for col in object_cols:
    #     df[col] = label_encoder.fit_transform(df[col])

    print("MIN_MOVIES = " + str(MIN_MOVIES) + ",  MIN_PEOPLE = " + str(MIN_PEOPLE))
    y = df.averageRating
    X = df.drop(['averageRating'], axis=1)

    models = []
    model_names = ["Ridge Regression", "Lasso Regression", "KNNeighbor", "Decision Tree Regressor",
                   "Random Forest Regressor", "XGBoost", "Neural Network"]

    from sklearn import linear_model
    RR_model = linear_model.RidgeCV()

    LR_model = linear_model.LassoCV()

    from sklearn.neighbors import KNeighborsRegressor
    n_neighbors = 5
    KNN_model = KNeighborsRegressor(n_neighbors=n_neighbors)

    from sklearn.tree import DecisionTreeRegressor
    random_state = 0
    DTR_model = DecisionTreeRegressor(random_state=random_state)

    from sklearn.ensemble import RandomForestRegressor
    n_estimators = 100
    RFR_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    models.append(RR_model)
    # models.append(LR_model)
    # models.append(KNN_model)
    # models.append(DTR_model)
    # models.append(RFR_model)
    # score_dataset_XGB(X, y)
    # score_dataset_ANN(X, y)

    i = 0
    for model in models:
        print(model_names[i] + " started")
        pipeline = Pipeline(steps=[('model', model)])
        scores = -1 * cross_val_score(pipeline, X, y,
                                      cv=5,
                                      scoring='neg_mean_absolute_error',
                                      n_jobs=5, verbose=True)
        i += 1
        print("MAE scores:\n", scores)
        print("Average: " + str(scores.mean()))


#   XGBOOST
def score_dataset_XGB(X, y):
    from xgboost import XGBRegressor
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    print("XGBRegressor started")
    n_estimators = 1000
    learning_rate = 0.05
    early_stopping_rounds = 5
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=5)
    # model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_set=[(X_valid, y_valid)], verbose=False)
    fit_params = {'early_stopping_rounds': 5,
                  'eval_metric': 'mae',
                  'verbose': False,
                  'eval_set': [[X_valid, y_valid]]}

    scores = cross_val_score(model, X_train, y_train,
                             cv=5,
                             scoring='neg_mean_absolute_error',
                             fit_params=fit_params)
    print("MAE scores:\n", scores)
    print("Average: " + str(scores.mean()))
    print(
        "n_estimators=" + str(n_estimators) + ", learning_rate=" + str(
            learning_rate) + ", early_stopping_rounds=" + str(early_stopping_rounds))


#   ARTIFICIAL NEURAL NETWORK
def score_dataset_ANN(X, y):
    import tensorflow as tf
    from tensorflow import keras
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
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


get_scores(pd.read_csv('IMDB files/imputed_movies_people_data.csv', header=0, low_memory=False, index_col=0))

winsound.Beep(1000, 800)
winsound.Beep(900, 500)
