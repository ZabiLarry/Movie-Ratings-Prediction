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
    counts = movies_data.averageRating.value_counts()
    counts = counts.sort_index(ascending=False)
    print(counts)
    counts.plot()
    plt.title('Distribution of ratings')
    plt.xlabel('Ratings')
    plt.ylabel('Movies')
    plt.xlim([1, 10])
    plt.show()


count_genres()


def count_people():
    principals_data = pd.read_csv('IMDB files/clean_principals_data.csv', header=0, low_memory=False, index_col=0)

    principals_data = principals_data.loc[principals_data['category'] == 'actor']
    count = principals_data.nconst.value_counts()
    count = count[count.values >= 10]
    principals_data = principals_data[
        principals_data.nconst.isin(count.index)]
    print(count)


def plotknn():
    maes = [[0.6591293378645158,
             0.6591722427782032,
             0.6594915755419315,
             0.6587484916735978,
             0.6556102767526443,
             0.6565936593236331,
             0.6603880000361789,
             0.6607255396114328,
             0.6632063058383942,
             0.6613777518335284,
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

    x = [0.05,
         0.06,
         0.1,
         0.12,
         0.15,
         0.18,
         0.24,
         0.3,
         0.36,
         0.42]
    labels = []

    for i in range(0, 1):
        plt.plot(x, maes[i])
    plt.title('MAE Vs ' + "learning_rate")
    plt.xlabel("learning_rate")
    plt.ylabel('MAE')
    # plt.legend()
    plt.show()


# plotknn()
# count_people()

# from sklearn.preprocessing import OneHotEncoder
# OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# principals_encoded = pd.DataFrame(OH_encoder.fit_transform(principals_data[['nconst']]), index=principals_data.index)
# OH_encoder.get_feature_names()
# principals_encoded.columns = OH_encoder.get_feature_names()
# print(principals_encoded.head())

def plotbar():
    x_values = ["Horror",
                "Documentary",
                "Drama",
                "Animation",
                "Action",
                "runtimeMinu",
                "Thriller",
                "Sci-Fi",
                "Biography",
                "CH",
                "XWW",
                "Comedy",
                "Crime",
                "startYear",
                "Musical",
                "IR",
                "Romance",
                "XEU",
                "HK",
                "HR",
                "TW",
                "GB",
                "RU",
                "PE",
                "uncommonReg",
                "BD",
                "TH",
                "IE",
                "IT",
                "Adventure",
                "Family",
                "NZ",
                "AT",
                "IN",
                "Fantasy",
                "RS",
                "US",
                "History",
                "CL",
                "DE",
                "UA",
                "UZ",
                "SI",
                "XWG",
                "CSHH",
                "TR",
                "LT",
                "ES",
                "NL",
                "VN",
                "Mystery",
                "PK",
                "Music",
                "FR"]
    importances = [0.29600000381469727,
                   0.12039999663829803,
                   0.05169999971985817,
                   0.03790000081062317,
                   0.027799999341368675,
                   0.02319999970495701,
                   0.01899999938905239,
                   0.01489999983459711,
                   0.01080000028014183,
                   0.010599999688565731,
                   0.010499999858438969,
                   0.009499999694526196,
                   0.009499999694526196,
                   0.009399999864399433,
                   0.008999999612569809,
                   0.008899999782443047,
                   0.007600000128149986,
                   0.007499999832361937,
                   0.007400000002235174,
                   0.007300000172108412,
                   0.007199999876320362,
                   0.007000000216066837,
                   0.007000000216066837,
                   0.006899999920278788,
                   0.006899999920278788,
                   0.006800000090152025,
                   0.006800000090152025,
                   0.0066999997943639755,
                   0.0066999997943639755,
                   0.006599999964237213,
                   0.006599999964237213,
                   0.006500000134110451,
                   0.006399999838322401,
                   0.006099999882280827,
                   0.006000000052154064,
                   0.005900000222027302,
                   0.005900000222027302,
                   0.005799999926239252,
                   0.005799999926239252,
                   0.005799999926239252,
                   0.005799999926239252,
                   0.00559999980032444,
                   0.005499999970197678,
                   0.005400000140070915,
                   0.005200000014156103,
                   0.005200000014156103,
                   0.005100000184029341,
                   0.004999999888241291,
                   0.004999999888241291,
                   0.004999999888241291,
                   0.004900000058114529,
                   0.004800000227987766,
                   0.004699999932199717,
                   0.004600000102072954]
    plt.bar(x_values, importances, orientation='vertical', color='r', edgecolor='k', linewidth=1.2)
    # Tick labels for x axis
    plt.xticks(x_values, x_values, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()


# plotbar()


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
