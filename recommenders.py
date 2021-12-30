import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Для работы с матрицами
from scipy.sparse import csr_matrix
from sklearn.pipeline import make_pipeline

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight

from abc import ABC, abstractmethod
import feature_generation

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import utils
from metrics import precision_at_k, recall_at_k
from utils import prefilter_items

# WARM_ITEM_ID = 999999
ITEM_COL = 'item_id'
USER_COL = 'user_id'
DEPARTMENT_COL = 'department'


class BaseRecommender(ABC):

    @abstractmethod
    def recommend(self, user, N=5, **kwargs):
        pass


class CollaborativeFilteringModels:
    def __init__(self, data, warm_start):

        self.warm_start = warm_start

        # TODO: weight matrix
        self.user_item_matrix = self.prepare_matrix(data)

        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

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

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

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

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)

        if self.warm_start:
            res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                        user_items=csr_matrix(
                                                                            self.user_item_matrix).tocsr(),
                                                                        N=N,
                                                                        filter_already_liked_items=False,
                                                                        recalculate_user=False)]
        else:
            res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                        user_items=csr_matrix(
                                                                            self.user_item_matrix).tocsr(),
                                                                        N=N,
                                                                        filter_already_liked_items=False,
                                                                        filter_items=[self.itemid_to_id[999999]],
                                                                        recalculate_user=False)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res


class ALSRecommendation(CollaborativeFilteringModels, BaseRecommender):
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
                 warm_start=True,
                 brand_name='Private',
                 use_bpr=False,
                 n_factors=100,
                 regularization=0.001,
                 iterations=15,
                 num_threads=4,
                 threshold=0.6,
                 random_state=42):

        super(ALSRecommendation, self).__init__(data=data, warm_start=warm_start)

        self.popular_items = PopularItemsRecommendation(warm_start).fit(data)
        self.threshold = threshold
        self.ctm_model = None

        # СТМ = товары под брендом Private
        self.ctm = item_features[
            item_features['brand'] == brand_name].item_id.unique()

        # TODO: other weigtht
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        if use_bpr:
            self.model = BayesianPersonalizedRanking(factors=n_factors,
                                                     regularization=regularization,
                                                     iterations=iterations,
                                                     num_threads=num_threads,
                                                     random_state=random_state)
        else:
            self.model = AlternatingLeastSquares(factors=n_factors,
                                                 regularization=regularization,
                                                 iterations=iterations,
                                                 num_threads=num_threads,
                                                 random_state=random_state)

    def fit(self):
        """Обучает ALS"""

        self.model.fit(csr_matrix(self.user_item_matrix).T.tocsr(), show_progress=False)

        return self.model

    def __get_items(self, user, n=5):
        return \
            self.popular_items[self.popular_items['user_id'] == user].sort_values('quantity', ascending=False).head(n)[
                'item_id'].tolist()

    def __similar_items_recommend(self, user, n=5):
        """Рекомендуем товары, похожие на топ-n купленных юзером товаров"""
        items = self.__get_items(user, n)

        top_rec = list()

        for item in items:
            recs = self.model.similar_items(self.itemid_to_id[item], N=2)  # самый похожий - это и есть сам товар
            top_rec.append(self.id_to_itemid[recs[1][0]])
        return top_rec

    def __similar_on_ctm_items_recommend(self, user, n=5):
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

                    if rec_item_with_prob[0][1] < self.threshold:
                        recs = self.model.similar_items(self.itemid_to_id[item], N=2)
                        top_rec.append(self.id_to_itemid[recs[1][0]])
                        break

                    top_rec.append(rec_item_with_prob[0][0])
                    break
        return top_rec

    def __similar_user_items_recommend(self, user, N=5, option=1):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        # Вариант 1: взять N users и с каждого самый популярный товар
        if option == 1:
            result = list()

            users_prob = self.model.similar_users(self.userid_to_id[user], N=N + 1)[1:]

            for user, _ in users_prob:
                result.append(self.popular_items[
                                  self.popular_items.user_id == self.id_to_userid[user]
                                  ].sort_values(by='quantity', ascending=False).head(1).item_id.tolist()[0]
                              )

            return result

        # Вариант 2: взять 1 users и у него N самых популярных товаров
        if option == 2:
            sim_user = self.model.similar_users(self.userid_to_id[user], N=2)[1][0]

            return self.popular_items[
                self.popular_items.user_id == self.id_to_userid[sim_user]
                ].sort_values(by='quantity', ascending=False).head(N).item_id.tolist()

        # assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        # return res

    def recommend(self, user, N=5, how=None):
        if how == 'similar_items':
            return self.__similar_items_recommend(user, n=N)

        if how == 'similar_on_ctm_items':
            return self.__similar_on_ctm_items_recommend(user, n=N)

        if how == 'similar_user_items':
            return self.__similar_user_items_recommend(user, N)

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)


class OwnRecommendation(CollaborativeFilteringModels, BaseRecommender):
    def __init__(self, data, warm_start):
        super(OwnRecommendation, self).__init__(data, warm_start=warm_start)

        # self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        self.own_recommender = None

    def recommend(self, user, N=5, **kwargs):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def fit(self):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        self.own_recommender = ItemItemRecommender(K=1, num_threads=4)
        self.own_recommender.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return self.own_recommender


class PopularItemsRecommendation(BaseRecommender):
    def __init__(self, warm_start):
        self.warm_start = warm_start
        self.popularity = None

    def recommend(self, user, N=5, **kwargs):
        return self.popularity.head(N)['item_id'].tolist()

    def fit(self, user_item_matrix: pd.DataFrame):
        """
        Return most popular user' items
        """
        popularity = user_item_matrix.groupby(by=['user_id', 'item_id']).agg({'quantity': 'count'}).reset_index()

        if self.warm_start:
            popularity = popularity[popularity['item_id'] != 999999]

        popularity.sort_values('quantity', ascending=False, inplace=True)

        self.popularity = popularity
        return popularity


class ClassificationModel:
    def __init__(self):
        self.model = None

    @staticmethod
    def create_user_item_matrix(user_prediction: pd.DataFrame, train_data: pd.DataFrame):
        df_items = user_prediction.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1,
                                                                                                            drop=True)
        df_items.name = 'item_id'
        user_prediction = user_prediction.drop('candidates', axis=1).join(df_items)

        df_ranker_train = train_data[[USER_COL, ITEM_COL]].copy()
        df_ranker_train['target'] = 1  # тут только покупки

        df_ranker_train = user_prediction.merge(df_ranker_train, on=[USER_COL, ITEM_COL], how='left')

        df_ranker_train['target'].fillna(0, inplace=True)
        return df_ranker_train

    @staticmethod
    def additional_features(to, item_features, user_features, data_train):
        additional_user_features = make_pipeline(
            feature_generation.AddPriceColumn(),
            feature_generation.AverageCheck(),
            feature_generation.AveragePriceCounItemADepartment(items_department=item_features[[ITEM_COL, DEPARTMENT_COL]]),
            feature_generation.PurchasesAMonth(),
            feature_generation.DropColumn(column_name='price'),
            feature_generation.UpdateFeaturesTable(feature_table=user_features,
                                X_old_columns_names=data_train.columns,
                                merge_on=[USER_COL])
        )

        user_features = additional_user_features.fit_transform(data_train)

        additional_items_features = make_pipeline(
            feature_generation.AddPriceColumn(),
            feature_generation.PurchasesCountAWeek(),
            feature_generation.AveragePurchasesCountADepartment(item_department_table=item_features[[ITEM_COL, DEPARTMENT_COL]]),
            feature_generation.LikePercentADepartment(),
            feature_generation.HigherThenMedianPricePercent(),
            feature_generation.UpdateFeaturesTable(feature_table=item_features,
                                X_old_columns_names=data_train.columns,
                                merge_on=ITEM_COL)
        )

        item_features = additional_items_features.fit_transform(X=data_train)

        to = to.merge(item_features, on='item_id', how='left')
        to = to.merge(user_features, on='user_id', how='left')

        return to

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.ranked_predict = X_train.copy()
        self.ranked_predict['proba_item_purchase'] = self.model.predict_proba(self.ranked_predict)[:,1]

    def recommend(self, user, N=5, **kwargs):
        return self.ranked_predict[self.ranked_predict[USER_COL] == user].sort_values('proba_item_purchase',
                                                                                      ascending=False).head(
            N).item_id.tolist()


class LightGBMRecommendation(ClassificationModel, BaseRecommender):
    def __init__(self,
                 **kwargs):

        super(LightGBMRecommendation, self).__init__()

        self.model = LGBMClassifier(**kwargs)
        # self.model = LGBMClassifier(boosting_type=boosting_type,
        #                             num_leaves=num_leaves,
        #                             max_depth=max_depth,
        #                             learning_rate=learning_rate,
        #                             n_estimators=n_estimators,
        #                             subsample_for_bin=subsample_for_bin,
        #                             objective=objective,
        #                             class_weight=class_weight,
        #                             min_split_gain=min_split_gain,
        #                             min_child_weight=min_child_weight,
        #                             min_child_samples=min_child_samples,
        #                             subsample=subsample,
        #                             subsample_freq=subsample_freq,
        #                             colsample_bytree=colsample_bytree,
        #                             reg_alpha=reg_alpha,
        #                             reg_lambda=reg_lambda,
        #                             random_state=random_state,
        #                             n_jobs=n_jobs,
        #                             # silent=silent,
        #                             importance_type=importance_type,
        #                             categorical_column=categorical_column,
        #                             min_data_in_leaf=min_data_in_leaf)

class CatBoostRecommendation(ClassificationModel, BaseRecommender):
   

    def __init__(self, depth=None, iterations=None, learning_rate=None, l2_leaf_reg=None, border_count=None,
                 bagging_temperature=None, random_strength=None, max_ctr_complexity=None, thread_count=4,
                 cat_features=None):
        super(CatBoostRecommendation, self).__init__()

        self.model = CatBoostClassifier(depth=depth, iterations=iterations, learning_rate=learning_rate,
                                        l2_leaf_reg=l2_leaf_reg, border_count=border_count,
                                        bagging_temperature=bagging_temperature, random_strength=random_strength,
                                        max_ctr_complexity=max_ctr_complexity, thread_count=thread_count,
                                        cat_features=cat_features)


from pathlib import Path

if __name__ == '__main__':
    pass
