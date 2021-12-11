import pandas as pd
import numpy as np


def prefilter_items(data,
                    items_data=None,
                    list_of_uninteresting_items=None,
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
    if list_of_uninteresting_items:
        unintersting_items = items_data.loc[
            items_data['department'].isin(list_of_uninteresting_items), 'item_id'].values
        data = data[~data['item_id'].isin(unintersting_items)]

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
            data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


def postfilter_items(user_id, recommednations):
    pass
