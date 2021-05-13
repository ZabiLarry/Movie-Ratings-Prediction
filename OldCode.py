import pandas as pd

#  pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 240)


def word_count():
    import collections
    import pandas as pd

    # Read input file, note the encoding is specified here
    # It may be different in your text file
    file = open('Thesis.txt', encoding="utf8")
    a = file.read()
    # Instantiate a dictionary, and for every word in the file,
    # Add to the dictionary if it doesn't exist. If it does, increase the count.
    wordcount = {}
    # To eliminate duplicates, remember to split by punctuation, and use case demiliters.
    for word in a.lower().split():
        word = word.replace(".", "")
        word = word.replace(",", "")
        word = word.replace(":", "")
        word = word.replace("\"", "")
        word = word.replace("!", "")
        word = word.replace("â€œ", "")
        word = word.replace("â€˜", "")
        word = word.replace("*", "")
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
    # Print most common word
    n_print = int(input("How many most common words to print: "))
    print("\nOK. The {} most common words are as follows\n".format(n_print))
    word_counter = collections.Counter(wordcount)
    for word, count in word_counter.most_common(n_print):
        print(word, ": ", count)
    # Close the file
    file.close()
    # Create a data frame of the most common words
    # Draw a bar chart
    lst = word_counter.most_common(n_print)
    df = pd.DataFrame(lst, columns=['Word', 'Count'])
    df.plot.bar(x='Word', y='Count')


word_count()


def encode_persons():
    # person_data.to_csv('IMDB files/alive_person_data.csv')
    # from sklearn.preprocessing import MultiLabelBinarizer
    # mlb = MultiLabelBinarizer()
    # a = mlb.fit_transform(person_data['primaryProfession'].str.split(','))
    # profession_encoded = pd.DataFrame(a, columns=mlb.classes_, index=person_data.index)
    # # print(profession_encoded)
    # # profession_frequency = profession_encoded.apply(pd.Series.value_counts, axis=0)
    # # print(profession_frequency)
    # encoded = pd.merge(person_data, profession_encoded, on="nconst", how="inner")
    # encoded.drop(encoded.columns[['primaryProfession', 'assistant', 'choreographer', 'electrical_department', 'legal',
    #                               'manager', 'production_department', 'publicist', 'talent_agent']], axis=1,
    #              inplace=True)
    # encoded.to_csv('IMDB files/encoded_person_data.csv')
    # return encoded
    pass


def score_dataset_SVR():
    from sklearn.metrics import mean_absolute_error
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('IMDB files/imputed_movies_people_data.csv', header=0, low_memory=False, index_col=0)
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    print("Categorical variables:" + str(object_cols))
    df = df.drop(object_cols, axis=1)

    y = df.averageRating
    X = df.drop(['averageRating'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()
    from sklearn import preprocessing
    print("Support Vector Regression started")
    kernel = "linear"
    # lab_enc = preprocessing.LabelEncoder()
    # encoded_train = lab_enc.fit_transform(y_train)
    # encoded_valid = lab_enc.fit_transform(y_valid)
    model = SVR(kernel=kernel, verbose=True)
    model.fit(label_X_train, y_train)
    preds = model.predict(label_X_valid)
    print("MAE SVR: " + str(mean_absolute_error(y_valid, preds)))
    print("kernel=" + kernel)

# score_dataset_SVR()
