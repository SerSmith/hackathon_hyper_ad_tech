from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

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