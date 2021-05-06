import pandas as pd
import winsound

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


def movies_data_reader():
    MIN_VOTES = 1000
    MIN_YEAR = 1970

    basics_data = pd.read_csv('IMDB files/basics_data.tsv', sep='\t', header=0, low_memory=False)
    movies_data_ratingless = basics_data[basics_data.titleType == 'movie']
    useless_cols = ['numVotes', 'titleType', 'originalTitle', 'endYear', 'isAdult']

    akas_data = pd.read_csv('IMDB files/akas_data.tsv', sep='\t', header=0, low_memory=False)
    movies_data_ratingless = pd.merge(movies_data_ratingless, akas_data, on="titleId", how="inner")  # (2137818, 16)
    useless_cols = useless_cols + ['attributes', 'types', 'isOriginalTitle', 'ordering', 'title', 'language']

    ratings_data = pd.read_csv('IMDB files/ratings_data.tsv', sep='\t', header=0, low_memory=False)
    raw_movies_data = pd.merge(movies_data_ratingless, ratings_data, on="titleId", how="inner")  # (1636341, 18)

    raw_movies_data.drop_duplicates(subset=["titleId"], inplace=True)  # (64863, 9)

    raw_movies_data['numVotes'] = raw_movies_data['numVotes'].apply(pd.to_numeric, errors='coerce')
    popular_movies_data = raw_movies_data[raw_movies_data['numVotes'] >= MIN_VOTES]
    print(popular_movies_data.shape)
    old_movies_data = popular_movies_data.loc[popular_movies_data['isAdult'] == '0']
    print(old_movies_data.shape)

    old_movies_data = old_movies_data.drop(useless_cols, axis=1)
    # print(old_movies_data)
    # old_movies_data.to_csv('IMDB files/movies_data_test.csv')

    old_movies_data['startYear'] = old_movies_data['startYear'].apply(pd.to_numeric, errors='coerce')
    movies_data = old_movies_data[old_movies_data['startYear'] >= MIN_YEAR]
    print(movies_data.shape)
    movies_data.reset_index(drop=True, inplace=True)
    print("movies cleaned")
    # movies_data.to_csv('IMDB files/clean_movies_data.csv')


def person_reader():
    person_data = pd.read_csv('IMDB files/person_data.tsv', sep='\t', header=0, low_memory=False, index_col=0)
    person_data = person_data.loc[person_data['deathYear'] == '\\N']
    person_data = person_data.loc[person_data['birthYear'] != '\\N']  # only born and alive people
    print("person cleaned")
    person_data.to_csv('IMDB files/clean_person_data.csv')


def cut_principals():
    person_data = pd.read_csv('IMDB files/clean_person_data.csv', header=0, low_memory=False, index_col=0)
    movies_data = pd.read_csv('IMDB files/clean_movies_data.csv', header=0, low_memory=False, index_col=1)
    principals_data = pd.read_csv('IMDB files/principals_data.tsv', sep='\t', header=0, low_memory=False, index_col=0)
    #  print(principals_data.shape)
    cut_principals_data = principals_data[
        principals_data.index.isin(movies_data.index)]  # drop people not in clean_movies
    cut_principals_data = cut_principals_data[
        cut_principals_data.nconst.isin(person_data.index)]  # drop people not in clean_person
    cut_principals_data.drop(cut_principals_data.columns[[0, 3]], axis=1, inplace=True)  # drop ordering and jobs(blank)
    print("principals cleaned and cut")
    cut_principals_data.to_csv('IMDB files/clean_principals_data.csv', index_label="titleId")


#  movies_data_reader()
#  person_reader()
#  cut_principals()

winsound.Beep(900, 500)
winsound.Beep(1000, 800)
print("have you NaNed?")


