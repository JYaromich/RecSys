import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight

from metrics import precision_at_k, recall_at_k
from utils import prefilter_items


class BaseRecommender:
    def __init__(self, data: pd.DataFrame, weighting=True, warm_start=True):
        self.warm_start = warm_start

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data, aggfunc='count'):
        """
        Method for transform input matrix to matrix like user-item matrix
        :param data: pandas.DataFrame input table witch must contain ext columns ['user_id', 'item_id', 'quantity']
        :return: user-item matrix
        """
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробоват ьдругие варианты
                                          aggfunc=aggfunc,
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    def fit(self, user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)

        return model


class MainRecommender(BaseRecommender):
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self,
                 data: pd.DataFrame,
                 item_features: pd.DataFrame,
                 weighting=True,
                 warm_start=True):

        super(MainRecommender, self).__init__(data, weighting, warm_start)

        self.popular_items = self.more_popular_items(item_features, data)

        self.ctm_model = None

    def more_popular_items(self, item_features: pd.DataFrame, data: pd.DataFrame, brand_name='Private'):
        """
        Return more popular user items

        :param user: user id
        :param brand_name: name of private brand
        :return: list of N item id
        """
        self.ctm = item_features[
            item_features['brand'] == brand_name].item_id.unique()  # СТМ = товары под брендом Private

        # наиболее популярные не СТМ товары
        popularity = data[~data['item_id'].isin(self.ctm)].groupby(['user_id', 'item_id'])[
            'quantity'].count().reset_index()

        popularity.sort_values('quantity', ascending=False, inplace=True)

        if self.warm_start:
            popularity = popularity[popularity['item_id'] != 999999]

        return popularity.sort_values('user_id', ascending=False)

    def __get_items(self, user, n=5):
        return \
            self.popular_items[self.popular_items['user_id'] == user].sort_values('quantity', ascending=False).head(n)[
                'item_id'].tolist()

    def get_similar_items_recommendation(self, user, n=5):
        """Рекомендуем товары, похожие на топ-n купленных юзером товаров"""
        items = self.__get_items(user, n)

        top_rec = list()

        for item in items:
            recs = self.model.similar_items(self.itemid_to_id[item], N=2)  # самый похожий - это и есть сам товар
            top_rec.append(self.id_to_itemid[recs[1][0]])
        return top_rec

    def get_similar_ctm_items_recommendation(self, user, n=5, threshold=0.6):
        """Рекомендуем товары, похожие на топ-n купленных юзером товаров из товаров собственного производства"""

        MAX_N = 100

        items = self.__get_items(user, n)

        top_rec = list()

        for item in items:
            start, add = 20, 1
            while start < MAX_N:
                start += add
                recs = self.model.similar_items(self.itemid_to_id[item], N=start)
                rec_items_with_prob = {self.id_to_itemid[item]: prob for item, prob in recs[1:]}
                rec_items = set(rec_items_with_prob.keys()) & set(self.ctm.tolist())

                if rec_items:
                    rec_items_with_prob = {item: rec_items_with_prob[item] for item in rec_items}

                    rec_item_with_prob = sorted(
                        rec_items_with_prob.items(),
                        key=lambda x: x[1],
                        reverse=False
                    )

                    if rec_item_with_prob[0][1] < threshold:
                        recs = self.model.similar_items(self.itemid_to_id[item], N=2)
                        top_rec.append(self.id_to_itemid[recs[1][0]])
                        break

                    top_rec.append(rec_item_with_prob[0][0])
                    break
        return top_rec

    def get_similar_users_recommendation(self, user, N=5, option=1):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        # Вариант 1: взять N users и с каждого самый популярный товар
        if option == 1:
            result = list()

            users_prob = self.model.similar_users(user, N=N + 1)[1:]

            for user, _ in users_prob:
                result.append(self.popular_items[
                                  self.popular_items.user_id == user
                              ].sort_values(by='quantity', ascending=False).head(1).item_id.tolist()[0]
                )

            return result

        # Вариант 2: взять 1 users и у него N самых популярных товаров
        if option == 2:
            sim_user = self.model.similar_users(user, N=2)[1][0]

            return self.popular_items[
                self.popular_items.user_id == sim_user
            ].sort_values(by='quantity', ascending=False).head(N).item_id.tolist()

        # assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        # return res


from pathlib import Path

if __name__ == '__main__':
    MAIN_PATH = Path('../')
    RETAIL_TRAIN_PATH = MAIN_PATH / 'retail_train.csv'
    PRODUCT_PATH = MAIN_PATH / 'product.csv'

    # retail data
    data = pd.read_csv(RETAIL_TRAIN_PATH)

    data.columns = [col.lower() for col in data.columns]
    data.rename(columns={'household_key': 'user_id',
                         'product_id': 'item_id'},
                inplace=True)

    test_size_weeks = 3

    # split on train and test
    data = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
    data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

    # items data
    item_features = pd.read_csv(PRODUCT_PATH)
    item_features.columns = [col.lower() for col in item_features.columns]
    item_features.rename(columns={'product_id': 'item_id'}, inplace=True)

    # result is table with columns user_id and actual purchases
    result = data_test.groupby('user_id')['item_id'].unique().reset_index()
    result.columns = ['user_id', 'actual']

    data = prefilter_items(data)

    rec_table = MainRecommender(data, item_features)
    print('similar items', rec_table.get_similar_items_recommendation(1060))
    print('similar ctm items', rec_table.get_similar_ctm_items_recommendation(1060))
    print('similar ctm items', rec_table.get_similar_users_recommendation(1060, option=1))
    print('similar ctm items', rec_table.get_similar_users_recommendation(1060, option=2))
