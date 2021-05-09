import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import winsound

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)


# df = pd.read_csv('IMDB files/imputed_movies_people_data.csv', header=0, low_memory=False, index_col=0)
#
# print(df.columns)
# s = (df.dtypes == 'object')
# object_cols = list(s[s].index)
# print("Categorical variables:" + str(object_cols))
# df = df.drop(object_cols, axis=1)
#
# y = df.averageRating
# X = df.drop(['averageRating'], axis=1)
#
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


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


def plotknn():
    maes = [[0.6766461071510804,
             0.6754891986506789,
             0.6747897210508917,
             0.6745956267424155,
             0.6743714418226444,
             0.6744044914246897,
             0.6746017747986113,
             0.6745909154741764,
             0.6747352451979216,
             0.6746690452369608,
                ],
            [0.7260893135257998, 0.7174919908921152, 0.7113548585454778, 0.7068117324178196, 0.7055370602788884,
             0.7100263141706341, 0.7096843015211116, 0.7137885187882198, 0.7134905785906418, 0.7148601773316094,
             0.7214766180648258],
            [0.7197819588535259, 0.7145331457710567, 0.7081853349745036, 0.704492702655178, 0.7033585370142432,
             0.7017232284156848, 0.706092843326885, 0.7060312994548976, 0.7080534552488138, 0.7130385088799014,
             0.7198083347986633],
            [0.878938815549419, 0.8786549085838249, 0.878444691927032, 0.8780960512752995, 0.8778222912282749,
             0.8773902856672819, 0.8773453173994549, 0.8772937139494327, 0.8776157770943331, 0.8783634817185953,
             0.8759476880467158]]

    x = [25,
         50,
         69,
         73,
         75,
         77,
         85,
         100,
         125,
         150]
    labels = ["mse", "friedman_mse", "mae", "poisson"]

    for i in range(0, 1):
        plt.plot(x, maes[i], label=labels[i])
    plt.title('MAE Vs ' + "n_estimators")
    plt.xlabel("n_estimators")
    plt.ylabel('MAE')
    plt.legend()
    plt.show()


plotknn()
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


def plot_test():
    df = pd.read_csv('IMDB files/imputed_movies_people_data.csv', header=0, low_memory=False, index_col=0)
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    print("Categorical variables:" + str(object_cols))
    df = df.drop(object_cols, axis=1)
    # label_encoder = LabelEncoder()
    # for col in object_cols:
    #     df[col] = label_encoder.fit_transform(df[col])
    y = df.averageRating
    X = df.drop(['averageRating'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    from sklearn.ensemble import RandomForestRegressor
    print("Random Forest Regressor started")
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    print("MAE:" + mae)
    # maes_list = []
    # changes = []
    # criteria = ["mse"]
    # first = True
    # for criterion in criteria:
    #     print("criterion: " + criterion)
    #     maes = []
    #     for i in range(1, 2):
    #         max_depth = 4 * i  # 22
    #         n_estimators = 100
    #         if first:
    #             changes.append(max_depth)
    #         model = RandomForestRegressor(
    #             # n_estimators=n_estimators, random_state=0, n_jobs=-2, max_depth=max_depth, criterion=criterion
    #         )
    #         model.fit(X_train, y_train)
    #         preds = model.predict(X_valid)
    #         mae = mean_absolute_error(y_valid, preds)
    #         print("MAE RFR: " + str(mae) + " max_depth=" + str(max_depth) + " n_estimators=" + str(n_estimators))
    #         maes.append(mae)
    #     first = False
    #     maes_list.append(maes)
    # for i in range(0, len(criteria)):
    #     plt.plot(changes, maes_list[i], label=criteria[i])
    # plt.title('MAE Vs ' + "max_depth")
    # plt.xlabel("max_depth")
    # plt.ylabel('MAE')
    # plt.legend()
    # plt.show()


# plot_test()

winsound.Beep(1000, 800)
winsound.Beep(900, 500)
