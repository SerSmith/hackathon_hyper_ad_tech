from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

def impute_column(data, column, data_test=None):

    data = data.copy()
    data = data.reset_index()

    if data_test is not None:
        data = data.append(data_test)

    mode_ = data[[column,	'bundle']].groupby('bundle').agg(pd.Series.mode).reset_index()
    mode_ = mode_.rename(columns={column: f"{column}_imputed"})
    mode_[f"{column}_imputed"] = mode_[f"{column}_imputed"].astype(str).replace({"[]": np.nan})

    data = data[['index', column, 'bundle']].merge(mode_, on='bundle', how='left').sort_values('index').reset_index(drop=True)

    return data[column].combine_first(data[f"{column}_imputed"]), mode_


def impute_column_test(data, column, mode_):
    data = data.copy()
    data = data.reset_index()
    data = data[['index', column, 'bundle']].merge(mode_, on='bundle', how='left').sort_values('index').reset_index(drop=True)

    return data[column].combine_first(data[f"{column}_imputed"])


def bow(tokenizer, text, bow_make_binary):
    print(text.shape)
    vectorizer = CountVectorizer(tokenizer=tokenizer, binary=bow_make_binary, lowercase=True, min_df=10)
    X = vectorizer.fit_transform(text.astype(str))
    bow = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())
    return bow


def bundle_preprocess(data, quantity_words_flag=True, bundles_bow_flag=True, bundles_parts_bow_flag=True, bow_make_binary=True):
    bundles = pd.DataFrame(data['bundle'].drop_duplicates().reset_index(drop=True), columns=['bundle'])

    if bundles_parts_bow_flag:
        bundles_parts_bow_features = bow(lambda x: x.split('.'), bundles['bundle'], bow_make_binary)   
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
    df_cities = df_cities.groupby(['region', 'settlement']).agg({'population':'sum', 'children':'sum', 'type': 'max', 'latitude_dd':'max','longitude_dd':'max'}).reset_index()
    df['city'] = df['city'].str.lower()
    df['oblast'] = df['oblast'].str.lower()
    df_cities['region'] = df_cities['region'].str.lower()
    df_cities['settlement'] = df_cities['settlement'].str.lower()
    df = pd.merge(df, df_cities, left_on = ['oblast','city'], right_on=['region','settlement'], how='left')
    df.drop(['region', 'settlement'], axis=1, inplace = True)

    # add timezone
    df.loc[df[df['shift'].isnull() == False].index, 'timezone'] = df[df['shift'].isnull() == False]['shift'].astype(str).apply(lambda x: x[3:5])
    df['timezone'] = df['timezone'].replace('', 0)
    df.loc[df[df['shift'].isnull() == False].index, 'timezone'] = df.loc[df[df['shift'].isnull() == False].index, 'timezone'].astype('int')
    return df 


def make_features_from_time(df,
                            dt_target='loc',
                            datetime_col_msk='created',
                            shift_col='shift',
                            fill_shift_na=False,
                            shift_filler='MSK',
                            dt_format='%Y-%m-%d %H:%M:%S'
                            ):
    """???????????????? ?????? ???? ?????????????? created

    Args:
        df ([type]): [description]
        dt_target (str, optional): [description]. Defaults to 'loc'.
        datetime_col_msk (str, optional): [description]. Defaults to 'created'.
        shift_col (str, optional): [description]. Defaults to 'shift'.
        fill_shift_na (bool, optional): [description]. Defaults to True.
        shift_filler (str, optional): [description]. Defaults to 'MSK'.
        dt_format (str, optional): [description]. Defaults to '%Y-%m-%d %H:%M:%S'.

    Returns:
        [type]: [description]
    """

    df = df.copy()

    prefix = 'msk_'
    shift = df[shift_col].copy()

    if dt_target == 'loc':
        prefix = 'loc_'
        if fill_shift_na:
            shift = shift.fillna(shift_filler)
        else:
            shift = shift.dropna()

    datetime_msk = pd.to_datetime(df.loc[shift.index, datetime_col_msk], format=dt_format)
    datetime_loc = datetime_msk.copy()

    if dt_target == 'loc':
        shift = shift.str.replace('MSK', '0')
        sign = shift.str.replace(r'[^+-]', '', regex=True)
        is_forward = sign == '+'
        shift_value = pd.to_timedelta(shift.str.replace(r'[^0-9]', '', regex=True).astype(int), 'H')
        datetime_loc[is_forward] += shift_value[is_forward]
        datetime_loc[~is_forward] -= shift_value[~is_forward]

    result = pd.DataFrame(dtype=object, index=df.index)

    minute = datetime_loc.dt.minute
    hour = datetime_loc.dt.hour
    weekday = datetime_loc.dt.weekday
    day = datetime_loc.dt.day
    month = datetime_loc.dt.month
    year = datetime_loc.dt.year

    academic_year = ((month <=5) | (month >=9)).astype(int)
    first_september = ((month == 9) & (day == 1)).astype(int)
    week_before_first_september = ((month == 8) & (day >= 25)).astype(int)

    early_morning = \
        ((hour >= 5) & (hour < 8)).astype(int)

    morning = \
        ((hour >= 8) & (hour < 11)).astype(int)

    day = \
        ((hour >= 11) & (hour < 18)).astype(int)
    
    evening = \
        ((hour >= 18) & (hour < 21)).astype(int)
    
    late_evening = \
        ((hour >= 21) & (hour <= 23)).astype(int)
    
    night = \
        ((hour >= 0) & (hour < 5)).astype(int)


    result[f'{prefix}minute'] = minute
    result[f'{prefix}hour'] = hour
    result[f'{prefix}day'] = day
    result[f'{prefix}month'] = month
    result[f'{prefix}year'] = year
    result[f'{prefix}weekday'] = weekday

    result[f'{prefix}is_weekend'] = (weekday.isin([5, 6])).astype(int)

    result[f'{prefix}days_to_weekend'] = 5 - result[f'{prefix}weekday']
    result[f'{prefix}days_to_weekend'].replace({-1:0}, inplace=True)

    
    result[f'{prefix}is_academic_year'] = academic_year
    result[f'{prefix}is_first_september'] = first_september
    result[f'{prefix}is_week_before_first_september'] = week_before_first_september

    result[f'{prefix}is_early_morning'] = early_morning
    result[f'{prefix}is_morning'] = morning
    result[f'{prefix}is_day'] = day
    result[f'{prefix}is_evening'] = evening
    result[f'{prefix}is_late_evening'] = late_evening
    result[f'{prefix}is_night'] = night

    bad_year = result[f'{prefix}year']==1970

    result.drop(columns=[f'{prefix}year'], inplace=True)

    for col in result.columns:
        result.loc[bad_year, col] = np.nan

    return result


def get_tags_from_time_features(df, tags_cols=None, tags_dict=None):
    """?????????????? ???????? ?????? ??????, ?????????????????????????????? ???? ????????

    Args:
        df ([type]): [description]
        tags_cols ([type], optional): [description]. Defaults to None.
        tags_dict ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    if 'loc_day' in df.columns:
        prefix = 'loc_'
    elif 'msc_day' in df.columns:
        prefix = 'msc_'
    else:
        '?? ???????????????? ?????? ?????? ??????????????. ?????????????????? ?????????????? ?????????????? ??reate_time_features'

    if tags_cols is None:
        tags_cols = [
            # 'month',
            'weekday',
            'is_weekend',
            'is_academic_year',
            # 'is_first_september',
            # 'is_week_before_first_september',
            'is_early_morning',
            'is_morning',
            'is_day',
            'is_evening',
            'is_late_evening',
            'is_night'
        ]

    if tags_dict is None:
        tags_dict = {
            'month': {7: '????????', 8: '????????????', 9: '????????????????'},
            'weekday': {
                0: '??????????????????????',
                1: '??????????????',
                2: '??????????',
                3: '??????????????',
                4: '??????????????',
                5: '??????????????',
                6: '??????????????????????'
                },
            'is_weekend': {0:'??????????????_????????', 1: '????????????????'},
            'is_academic_year': {0: '????????????????', 1: '??????????????_??????'},
            'is_first_september': {0: '', 1: '????????????_????????????????'},
            'is_week_before_first_september': {0: '', 1: '????????????_??????????_??????????????_??????????'},
            'is_early_morning': {0: '', 1: '????????????_????????'},
            'is_morning': {0: '', 1: '????????'},
            'is_day': {0: '', 1: '????????'},
            'is_evening': {0: '', 1: '??????????'},
            'is_late_evening': {0: '', 1: '??????????????_??????????'},
            'is_night': {0: '', 1: '????????'}
        }

    tags_cols_prefix = [f'{prefix}{col}' for col in tags_cols]

    tags_df = df[tags_cols_prefix].copy()
    tags_df = tags_df.fillna('')

    for col, col_prefix in zip(tags_cols, tags_cols_prefix):
        tags_df[col_prefix] = tags_df[col_prefix].replace(tags_dict[col])

    time_of_day_columns = \
        [
            'is_early_morning',
            'is_morning',
            'is_day',
            'is_evening',
            'is_late_evening',
            'is_night'
        ]

    time_of_day_columns_prefix = [f'{prefix}{col}' for col in time_of_day_columns]

    tags_df[f'{prefix}time_of_day'] = tags_df[time_of_day_columns_prefix].sum(axis=1)

    tags_df.drop(columns=time_of_day_columns_prefix, inplace=True)

    tags_df.columns = [col + '_tag' for col in tags_df.columns]

    return tags_df


def get_size_city(row):
    if pd.isnull(row['population']):
        return 'unknown city'
    if row['population'] > 5000000:
        size = 'very big city'
    elif row['population'] > 1000000:
        size = 'big city'
    elif row['population'] > 60000:
        size = 'medium city'
    elif row['population'] <= 60000:
        size = 'small city'
    return size


def get_tags_from_cities_features(df):
    """
    ?????????????? ???????? ???? ??????, ?????????????????????????????? ???? ?????????????? ?? ???????????????? ??????????
    """
    df = df.copy()
    df['type_city'] = df['type']
    df['size_city'] = df.apply(get_size_city, axis=1)
    tags_df = df[['type_city', 'size_city', 'timezone']]
    return tags_df

def get_version_float(x):
    x_list = str(x).split('.')
    
    try:
        if len(x_list) >=2:
            out = int(x_list[0]) + 0.01 *int(x_list[1])
        else:
            out = int(x_list[0])
    except ValueError:
        out = -10
    return out


def phone_tags(data):
    out = data[['os']]
    out['os'] = out['os'].str.upper()
    out['osv_num'] = data['osv'].astype(str).apply(get_version_float)
    out['new_phone'] = np.nan

    out.loc[(out['os'] == 'ANDROID') & (out['osv_num'] >= 9), 'new_phone'] = 'new_phone'
    out.loc[((out['os'] == 'IOS') & (out['osv_num'] >= 10.03)), 'new_phone'] = 'new_phone'

    out.loc[((out['os'] == 'ANDROID') & (out['osv_num'] < 9)) & ((out['os'] == 'ANDROID') & (out['osv_num'] >= 7)), 'new_phone'] = 'medium_phone'
    out.loc[((out['os'] == 'IOS') & (out['osv_num'] < 10.03)) & ((out['os'] == 'ANDROID') & (out['osv_num'] >= 10)), 'new_phone'] = 'medium_phone'

    out.loc[((out['os'] == 'ANDROID') & (out['osv_num'] < 7) ), 'new_phone'] = 'old_phone'
    out.loc[((out['os'] == 'IOS') & (out['osv_num'] < 10)), 'new_phone'] = 'old_phone'

    out = out.drop(columns=['osv_num'])
    out = out.rename(columns={'os': 'os_'})
    return out

def create_description(data, tags_list):
    out = data[tags_list[0]]
    for column in tags_list[1: ]:
        print(column)
        print(out.shape)
        out = out.astype(str) + '.tag: '+ data[column].astype(str) +  f' tag_t: {column}'
    return out