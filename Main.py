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
folder = "New IMDB files"
# folder = "IMDB files"
print("have you NaNed?")

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


def add_people(movies_data):
    print("started adding people")
    principals_data = pd.read_csv(folder + '/clean_principals_data.csv', header=0, low_memory=False, index_col=0)

    # principals_data = principals_data.loc[principals_data['category'] == 'director']
    count = principals_data.nconst.value_counts()
    count = count[count.values >= MIN_PEOPLE]
    principals_data = principals_data[
        principals_data.nconst.isin(count.index)]  # Remove artists with less than MIN_PEOPLE
    from sklearn.preprocessing import OneHotEncoder
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    principals_encoded = pd.DataFrame(OH_encoder.fit_transform(principals_data[['nconst']]),
                                      index=principals_data.index)
    # OH_encoder.get_feature_names()
    principals_encoded.columns = OH_encoder.get_feature_names()
    principals_encoded = principals_encoded.reset_index().groupby('titleId').max()
    # principals_encoded.drop_duplicates(subset=principals_data.columns[[0]], inplace=True)
    movies_people_data = pd.merge(movies_data, principals_encoded, on="titleId", how="left")
    imputer = SimpleImputer(missing_values=np.NaN, strategy='constant', fill_value=0.0)
    imputed_movies_people_data = pd.DataFrame(imputer.fit_transform(movies_people_data))
    imputed_movies_people_data.columns = movies_people_data.columns
    imputed_movies_people_data.index = movies_people_data.index
    imputed_movies_people_data.to_csv(folder + '/imputed_movies_people_data.csv')
    print("people added")
    return imputed_movies_people_data


def reset():
    movies_data = pd.read_csv(folder + '/clean_movies_data.csv', header=0, low_memory=False, index_col=1)
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


# INFORMATION GATHERING
def check_importance(model, columns):
    importances = list(model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(columns, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    beep()
    plt.bar(x_values, importances, orientation='vertical', color='r', edgecolor='k', linewidth=1.2)
    # Tick labels for x axis
    plt.xticks(x_values, columns, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()


def beep():
    winsound.Beep(1000, 800)
    winsound.Beep(900, 500)


def line_plot(x, mae, x_label):
    beep()
    plt.plot(x, mae)
    plt.title('MAE Vs ' + x_label)
    plt.xlabel(x_label)
    plt.ylabel('MAE')
    plt.show()


def lines_plot(x, mae, labels, x_label):
    beep()
    for i in range(0, len(labels)):
        plt.plot(x, mae[i], label=labels[i])
    plt.title('MAE Vs ' + x_label)
    plt.xlabel(x_label)
    plt.ylabel('MAE')
    plt.legend()
    plt.show()


# ---- ALGORITHMS ----
def score_dataset_RR(X_train, X_valid, y_train, y_valid):
    from sklearn import linear_model
    print("Ridge Regression started")  # default: 0.66576509
    maes = []
    changes = []
    for i in range(1, 13):
        alphas = 10.1  # MAE RR: 0.6643688873294995 alpha=10.1
        changes.append(alphas)
        model = linear_model.RidgeCV(alphas=alphas)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, preds)
        maes.append(mae)
        print("MAE RR: " + str(mae) + " alpha=" + str(alphas))
        log.write("\n MAE RR: " + str(mae) + " alpha=" + str(alphas))

    line_plot(changes, maes, "alphas")


def score_dataset_LR(X_train, X_valid, y_train, y_valid):
    from sklearn import linear_model
    print("Lasso Regression started")
    maes = []
    changes = []
    for i in range(1, 13):
        alphas = 1e3
        max_iter = 10
        changes.append(alphas)
        model = linear_model.LassoCV(alphas=alphas, max_iter=max_iter, normalize=True)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, preds)
        maes.append(mae)
        print("MAE LR: " + str(mae) + " alphas:" + str(alphas) + " max_iter:" + str(max_iter))
        log.write("\n MAE LR: " + str(mae) + " alphas:" + str(alphas) + " max_iter:" + str(max_iter))
    line_plot(changes, maes, "n_alphas")


def score_dataset_KNN(X_train, X_valid, y_train, y_valid):
    from sklearn.neighbors import KNeighborsRegressor
    print("KNNeighbor started")
    maes_list = []
    changes = []
    weights = ['uniform', 'distance']
    ps = [2, 1]
    for p in ps:
        for weight in weights:
            maes = []
            for i in range(1, 12):
                n_neighbors = i * 3
                changes.append(n_neighbors)
                knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weight, p=p)
                knn_model.fit(X_train, y_train)
                preds = knn_model.predict(X_valid)
                mae = mean_absolute_error(y_valid, preds)
                maes.append(maes)
                print("MAE KNN: " + str(mae) + " n_neighbors=" + str(n_neighbors) + " weights: " + weight + " p:" + str(
                    p))
                log.write(
                    "\n MAE KNN: " + str(mae) + " n_neighbors=" + str(
                        n_neighbors) + " weights: " + weight + " p:" + str(p))
            maes_list.append(maes)
    labels = ["weight: uniform, p: Euclidean", "weight: distance, p: Euclidean", "weight: uniform, p: Manhattan",
              "weight: distance, p: Manhattan"]
    lines_plot(changes, maes_list, labels, "neighbors")


def score_dataset_DTR(X_train, X_valid, y_train, y_valid):
    from sklearn.tree import DecisionTreeRegressor
    print("Decision Tree Regressor started")
    maes_list = []
    changes = []
    criteria = [
        # "mse", "friedman_mse", "mae",
        "poisson"
    ]
    for criterion in criteria:
        maes = []
        for i in range(1, 12):
            max_depth = i * 12
            changes.append(max_depth)
            model = DecisionTreeRegressor(random_state=0, criterion=criterion)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            mae = mean_absolute_error(y_valid, preds)
            maes.append(mae)
            print(
                "MAE DTR: " + str(mae) + " max_depth: " + str(max_depth) + " criterion:" + criterion + " depth:" + str(
                    model.get_depth()) + " / leaves:" + str(model.get_n_leaves()))
            log.write("\n MAE DTR: " + str(mae) + " max_depth: " + str(
                max_depth) + " criterion:" + criterion + " depth:" + str(model.get_depth()) + " / leaves:" + str(
                model.get_n_leaves()))
        maes_list.append(maes)

    lines_plot(changes, maes_list, criteria, "max_depth")


def score_dataset_RFR(X_train, X_valid, y_train, y_valid):
    from sklearn.ensemble import RandomForestRegressor
    print("Random Forest Regressor started")
    maes_list = []
    changes = []
    criteria = ["mse"]
    first = True
    for criterion in criteria:
        print("criterion: " + criterion)
        maes = []
        for i in range(1, 7):
            max_depth = 27  # 22
            n_estimators = 85
            if first:
                changes.append(n_estimators)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=-2, max_depth=max_depth,
                                          criterion=criterion)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            mae = mean_absolute_error(y_valid, preds)
            maes.append(mae)
            print("MAE RFR: " + str(mae) + " max_depth=" + str(max_depth) + " n_estimators=" + str(n_estimators))
            log.write("\n MAE RFR: " + str(mae) + " max_depth=" + str(max_depth) + " n_estimators=" + str(n_estimators))
        first = False
        maes_list.append(maes)
    lines_plot(changes, maes_list, criteria, "n_estimators")
    # check_importance(model, X_train.columns)


def score_dataset_XGB(X_train, X_valid, y_train, y_valid):
    from xgboost import XGBRegressor
    import pickle
    print("XGBRegressor started")
    maes = []
    changes = []
    n_estimators = 800
    n_jobs = 4
    early_stopping_rounds = 5
    print("early_stopping_rounds=" + str(early_stopping_rounds))
    for i in range(1, 2):
        learning_rate = .15
        changes.append(learning_rate)
        my_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=n_jobs)
        my_model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_set=[(X_valid, y_valid)],
                     verbose=False)
        preds = my_model.predict(X_valid)
        mae = mean_absolute_error(y_valid, preds)
        maes.append(mae)
        print("MAE XGB: " + str(mae) + " n_estimators=" + str(my_model.n_estimators) + ", learning_rate=" + str(
            learning_rate) + ", early_stopping_rounds=" + str(early_stopping_rounds))
        log.write("\n MAE XGB: " + str(mae) + " n_estimators=" + str(my_model.n_estimators) + ", learning_rate=" + str(
            learning_rate) + ", early_stopping_rounds=" + str(early_stopping_rounds))
        check_importance(my_model, X_train.columns)
        pickle_name = "XGBoost Model"
        pickle.dump(my_model, open(pickle_name, 'wb'))
        print("pickle: " + pickle_name)
    # line_plot(changes, maes, "learning_rate")


def score_XGB_hyper(X_train, X_valid, y_train, y_valid):
    print("Hyper X started")
    from xgboost import XGBRegressor
    # from hyperopt import hp
    from sklearn.model_selection import GridSearchCV
    param_tuning = {
        'learning_rate': [0.1, 0.15],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators': [200, 500],
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator=xgb_model,
                           param_grid=param_tuning,
                           scoring='neg_mean_absolute_error',  # MAE
                           # scoring = 'neg_mean_squared_error',  #MSE
                           cv=5,
                           n_jobs=-2,
                           verbose=1)

    gsearch.fit(X_train, y_train)

    print(gsearch.best_params_)


def score_dataset_ANN(X_train, X_valid, y_train, y_valid):
    import tensorflow as tf
    from tensorflow import keras
    print("Artificial Neural Network started")
    epochs = 100
    patience = 5
    loss = 'mean_squared_error'
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=[len(X_train.keys())]),  # input layer
        keras.layers.Dense(32, activation='relu'),  # hidden layer (1)
        keras.layers.Dense(32, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(32, activation='relu'),  # hidden layer (3)
        keras.layers.Dense(1)  # output layer
    ])
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, callbacks=[callback])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    preds = model.predict(X_valid)
    print("MAE ANN: " + str(mean_absolute_error(y_valid, preds)))
    print(" epochs=" + str(epochs) + " / loss:" + loss)
    log.write("\n MAE ANN: " + str(mean_absolute_error(y_valid, preds)) +
              "\n epochs=" + str(epochs))


def get_scores(df):
    from sklearn.model_selection import train_test_split
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

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # score_dataset_RR(X_train, X_valid, y_train, y_valid)
    # score_dataset_LR(X_train, X_valid, y_train, y_valid)
    # score_dataset_KNN(X_train, X_valid, y_train, y_valid)
    # score_dataset_DTR(X_train, X_valid, y_train, y_valid)
    # score_dataset_RFR(X_train, X_valid, y_train, y_valid)
    score_dataset_XGB(X_train, X_valid, y_train, y_valid)
    # score_XGB_hyper(X_train, X_valid, y_train, y_valid)
    # score_dataset_ANN(X_train, X_valid, y_train, y_valid)


reset()
# print(movies_data.isnull().sum())
# get_scores(pd.read_csv(folder + '/encoded_movies_data.csv', header=0, low_memory=False, index_col=0))

# get_scores(pd.read_csv(folder + '/imputed_movies_people_data.csv', header=0, low_memory=False, index_col=0))

beep()
