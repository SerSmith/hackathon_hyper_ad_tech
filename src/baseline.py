import pandas as pd

def find_prob_bundle(df):
    """
    Находим вероятности принадлежности к сегменту только на основе bundle

    на выходе df с колонками 1,2,3,4,5 в которых лежат вероятности принадлежности к соответствующему сегменту
    """
    d = df.groupby(['Segment','bundle']).count()
    d = d.reset_index()
    d_sum = d.groupby(['bundle']).agg({'created':'sum'}).reset_index()
    d_sum.columns = ['bundle','sum_cnt_segment']
    d = pd.merge(d, d_sum, on='bundle', how='inner')
    d['part_segment'] = d['created']/d['sum_cnt_segment']
    d = d.pivot(index='bundle', columns='Segment', values='part_segment').fillna(0).reset_index()
    return d

def train_and_predict(train, test):
    print('train')
    bundle_procent_segm = find_prob_bundle(train)
    print('predict')
    prediction = pd.merge(test, bundle_procent_segm, on ='bundle', how='left')
    return prediction