import pandas as pd
import numpy as np
import requests

retail_data_convert = {
    'actual': lambda x: list(map(int, x[1:-1].strip().split())),
    'random_recommendation': lambda x: list(map(int, x[1:-1].strip().split(','))),
    'popular_recommendation': lambda x: list(map(int, x[1:-1].strip().split(','))),
    'itemitem': lambda x: list(map(int, x[1:-1].strip().split(','))),
    'cosine': lambda x: list(map(int, x[1:-1].strip().split(','))),
    'tfidf': lambda x: list(map(int, x[1:-1].strip().split(','))),
    'own_purchases': lambda x: [] if x == '[]' else list(
        map(int, re.sub(r"\s+", "", x[1:-1], flags=re.UNICODE).split(',')))
}


def send_tg_message(text='Cell execution completed.'):
    tg_api_token = '5043397400:AAExpKcTarTWA2nYlvB_rOyL7vMVe8e1DFw'
    tg_chat_id = '1971178318'
    
    requests.post(
        'https://api.telegram.org/' +
        'bot{}/sendMessage'.format(tg_api_token), 
        params=dict(chat_id=tg_chat_id, text=text)
    )

#wrapper for prefilter func
class PrefilterItems:
    def __init__(self, items_data, take_n_popular, warm_start, main_data=None) -> None:
        self.main_data = main_data
        self.items_data = items_data
        self.take_n_popular =take_n_popular
        self.warm_start = warm_start
    
    def __call__(self, *args: 'Any', **kwds: 'Any') -> pd.DataFrame:
        params = {
            'data': self.main_data,
            'items_data': self.items_data,
            'take_n_popular': self.take_n_popular,
            'warm_start': self.warm_start
        }
        
        return prefilter_items(**params)
    


def prefilter_items(data,
                    items_data=None,
                    min_price_4_filter=.9,
                    max_price_4_filter=200,
                    take_n_popular=5000,
                    warm_start=True):
    """
    Function is filtering input data. It must contains next column (items_id, user_id, week_no (count of weeks when
    company was started sales), quantity (count of purchases).
    sales_value (price for item))

    :param data: pandas.DataFrame with retail data.
    :param items_data: pandas.DataFrame with items descriptions. It must contains columns department witch has sense as
    type of purchases. Default=None
    :param list_of_uninteresting_items: list witch contains name of uninteresting department
    :param min_price_4_filter: items which have a price less will removed
    :param max_price_4_filter: items which have a price more will removed
    :param take_n_popular: take the most popular items
    :param warm_start: if True, items witch doesn't get in top `take_n_popular` will set items_id = 999999
    :return: filtering data
    """

    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[
        popularity['share_unique_users'] > 0.5].item_id.tolist()  # товары о которые покупали более 50% users
    data = data[~data['item_id'].isin(top_popular)]  # выбираем не популярные товары

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_not_popular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_not_popular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    month_without_sales = 12
    weeks_a_month = 4

    time_filter = data['week_no'] < data['week_no'].max() - month_without_sales * weeks_a_month
    quntity_filter = data['quantity'] == 0
    data = data[~(time_filter & quntity_filter)]

    # Уберем не интересные для рекоммендаций категории (department)
    if items_data is not None:
        department_size = pd.DataFrame(items_data. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = items_data[
            items_data['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    # Цены указаны в долларах. Будем убирать товары дешевле 0,9 долларов
    filter_ = data['quantity'] > 1

    data['price_for_item'] = data['sales_value']
    data.loc[filter_, 'price_for_item'] = data.loc[filter_, 'price_for_item'] / data.loc[filter_, 'quantity']
    data = data[data['price_for_item'] > min_price_4_filter]

    # Уберем слишком дорогие товары
    data.loc[data['price_for_item'] > max_price_4_filter].drop('price_for_item', axis=1)

    if take_n_popular:
        # Возмем топ по популярности
        popularity = data.groupby('item_id')['quantity'].sum().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

        top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

        # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
        if warm_start:
            # Выбираем топ take_n_popular товары и пользователей которые покупали эти товары. Остальные удаляем.
            data = data[data['item_id'].isin(top)]
        else:
            data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

        # data = data[data.item_id.isin(top)]
    return data


def postfilter_items(user_id, recommednations):
    pass


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df