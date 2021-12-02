from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def impute_column(data, column):

    data = data.copy()
    data = data.reset_index()

    mode_ = data[[column,	'bundle']].groupby('bundle').agg(pd.Series.mode).reset_index()
    mode_ = mode_.rename(columns={column: f"{column}_imputed"})
    mode_[f"{column}_imputed"] = mode_[f"{column}_imputed"].astype(str).replace({"[]": np.nan})

    data = data[['index', column, 'bundle']].merge(mode_, on='bundle').sort_values('index')

    return data[column].combine_first(data[f"{column}_imputed"])


def bow(tokenizer, text):
    print(text.shape)
    vectorizer = CountVectorizer(tokenizer=tokenizer, binary=True, lowercase=True, min_df=10)
    X = vectorizer.fit_transform(text.astype(str))
    bow = pd.DataFrame(X.todense(), columns=list(vectorizer.vocabulary_.keys()))
    return bow


def bundle_preprocess(data, quantity_words_flag=True, bundles_bow_flag=True, bundles_parts_bow_flag=True):
    bundles = pd.DataFrame(data['bundle'].drop_duplicates().reset_index(drop=True), columns=['bundle'])

    if bundles_parts_bow_flag:
        bundles_parts_bow_features = bow(lambda x: x.split('.'), bundles['bundle'])   
        bundles = pd.concat([bundles, bundles_parts_bow_features], axis=1)
        print('bundles_parts_bow_flag ready')


    if bundles_bow_flag:
        bundles_bow_features = pd.get_dummies(bundles['bundle'])
        bundles = pd.concat([bundles, bundles_bow_features], axis=1)
        print('bundles_bow_flag ready')
    
    if quantity_words_flag:
        bundles['quanitity_points'] = bundles['bundle'].str.count('\.')
        print('quantity_words_flag ready')


    # return bundles, bundles_parts_bow_features, bundles_bow_features
    return bundles


def make_features_from_cities(df, path_cities):
    df_cities = pd.read_csv(path_cities)
    df_cities = df_cities.groupby(['region','settlement']).agg({'population':'sum', 'children':'sum', 'type': 'max', 'latitude_dd':'max','longitude_dd':'max'}).reset_index()
    df['city'] = df['city'].str.lower()
    df['oblast'] = df['oblast'].str.lower()
    df_cities['region'] = df_cities['region'].str.lower()
    df_cities['settlement'] = df_cities['settlement'].str.lower()
    df = pd.merge(df, df_cities, left_on = ['oblast','city'], right_on=['region','settlement'], how='left')
    df.drop(['region','settlement'], axis=1, inplace = True)

    # add timezone
    df.loc[df[df['shift'].isnull()==False].index,'timezone'] = df[df['shift'].isnull()==False]['shift'].astype(str).apply(lambda x: x[3:5])
    df['timezone'] = df['timezone'].replace('', 0)
    df.loc[df[df['shift'].isnull()==False].index, 'timezone'] = df.loc[df[df['shift'].isnull()==False].index, 'timezone'].astype('int')
    return df 