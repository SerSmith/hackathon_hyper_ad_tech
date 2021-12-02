from sklearn.metrics import roc_auc_score


def check_roc_auc(prediction):
    # предполагается, что предсказания в колонках с названиями 1,2,3,4,5
    # пропуски в предсказаниях заполняются нулями
    for i in range(1,6):
        print(f'cnt prediction {i} contains NaN {prediction[prediction[i].isnull()].shape}')
        prediction[i] = prediction[i].fillna(0)
    for i in range(1,6):
        prediction[f'Segment_{i}'] = prediction['Segment'].apply(lambda x: 1 if x==i else 0)
        print(roc_auc_score(prediction[f'Segment_{i}'], prediction[i]))