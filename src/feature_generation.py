import pprint

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline

import pandas as pd


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column_names]


class AddPriceColumn(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['price'] = X['sales_value']

        filter_ = X['quantity'] > 1
        X.loc[filter_, 'price'] = X.loc[filter_, 'price'] / X.loc[filter_, 'quantity']

        return X


class AverageCheck(BaseEstimator, TransformerMixin):
    """Средний чек"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        user_avg_check = X.groupby(by='user_id').agg({'price': 'mean'}).reset_index()
        user_avg_check.rename(columns={'price': 'avg_check'}, inplace=True)
        return X.merge(user_avg_check, on='user_id', how='left')


class AveragePriceCounItemADepartment(BaseEstimator, TransformerMixin):
    """
    1) Средняя сумма покупки 1 товара в каждой категории (55 руб для категории молоко, 230 руб для категории мясо, ...)
    2) Количество покупок в каждой категории
    """

    def __init__(self, items_department: pd.DataFrame):
        self.items_department = items_department

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_department = X.copy()
        X_department = X.merge(self.items_department, on='item_id', how='left')
        average_item_a_department = X_department.groupby(by=['user_id', 'department']).agg(
            {'price': 'mean'}).reset_index()

        count_item_a_department = X_department.groupby(by=['user_id', 'department']).agg(
            {'quantity': 'count'}).reset_index()

        average_item_a_department_matrix = pd.pivot_table(data=average_item_a_department, index='user_id',
                                                          columns='department', values='price',
                                                          fill_value=0).reset_index()

        count_item_a_department = pd.pivot_table(data=count_item_a_department, index='user_id',
                                                 columns='department', values='quantity',
                                                 fill_value=0).reset_index()

        # average_item_a_department_matrix.drop([' '], axis=1, inplace=True)

        X = X.merge(average_item_a_department_matrix, on='user_id', how='left')
        X = X.merge(count_item_a_department, on='user_id', how='left', suffixes=('_price', '_count'))

        return X


class PurchasesAMonth(BaseEstimator, TransformerMixin):
    """Частотность покупок"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['month_no'] = X['week_no'] // 4
        frequency_a_month = X.groupby(by=['user_id', 'month_no']).agg({'quantity': 'sum'}).reset_index()
        frequency_a_month.rename(columns={'quantity': 'quantity_per_month'}, inplace=True)
        return X.merge(frequency_a_month, on=['user_id', 'month_no'], how='left').drop('month_no', axis=1)


class UpdateFeaturesTable(BaseEstimator, TransformerMixin):
    def __init__(self, feature_table, X_old_columns_names, merge_on):
        self.feature_table = feature_table

        if type(X_old_columns_names) != list:
            X_old_columns_names = list(X_old_columns_names)

        self.X_old_columns_names = X_old_columns_names

        if type(merge_on) != list:
            merge_on = [merge_on]

        self.merge_on = merge_on

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        new_columns = list(set(X.columns) - set(self.X_old_columns_names))
        new_columns.extend(self.merge_on)

        new_features = X[new_columns]
        new_features = new_features.groupby(by=self.merge_on).agg(
            {column: 'mean' for column in new_columns if column not in self.merge_on}
        ).reset_index()

        # return self.feature_table.merge(new_features, on=self.merge_on, how='left')
        return new_features.merge(self.feature_table, on=self.merge_on, how='left')


class PurchasesCountAWeek(BaseEstimator, TransformerMixin):
    """Среднее кол-во покупок в неделю"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        frequency_purchases_a_week = X.groupby(by=['item_id', 'week_no']).agg({'quantity': 'sum'}).reset_index()
        avg_frequency_purchases_a_week = frequency_purchases_a_week.groupby(by='item_id').agg(
            {'quantity': 'mean'}
        ).reset_index()

        avg_frequency_purchases_a_week.rename(columns={'quantity': 'avg_quantity_week'}, inplace=True)

        return X.merge(avg_frequency_purchases_a_week, on='item_id', how='left')


class AveragePurchasesCountADepartment(BaseEstimator, TransformerMixin):
    """Среднее кол-во покупок 1 товара в категории в неделю"""

    def __init__(self, item_department_table):
        self.item_department_table = item_department_table

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_department = X.copy().merge(self.item_department_table, on='item_id', how='left')

        frequency_purchases_a_department = X_department.groupby(by=['department', 'week_no']).agg(
            {'quantity': 'sum'}
        ).reset_index().groupby(by=['department']).agg({'quantity': 'mean'}).reset_index()

        frequency_purchases_a_department.rename(columns={'quantity': 'popular_department'}, inplace=True)

        return X.merge(X_department.merge(frequency_purchases_a_department, on='department', how='left'))


class LikePercentADepartment(BaseEstimator, TransformerMixin):
    """(Кол-во покупок в неделю) / (Среднее кол-во покупок 1 товара в категории в неделю)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        popular_items_a_week = X.groupby(by=['item_id', 'week_no']).agg({'quantity': 'sum'}).reset_index().groupby(
            by='item_id').agg({'quantity': 'mean'}).reset_index()

        popular_items_a_week.rename(columns={'quantity': 'popular_items_a_week'}, inplace=True)
        X = X.merge(popular_items_a_week, on='item_id', how='left')

        X['like_percent_a_department'] = X.popular_items_a_week / X.popular_department * 100

        return X.fillna(0)


class HigherThenMedianPricePercent(BaseEstimator, TransformerMixin):
    """Цена / Средняя цена товара в категории"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        avg_price_a_department = X.groupby(by='department').agg({'price': 'mean'}).reset_index()
        avg_price_a_department.rename(columns={'price': 'avg_price_a_department'}, inplace=True)
        X = X.merge(avg_price_a_department, on='department', how='left')
        X['higher_then_median_price_percent'] = X.price / X.avg_price_a_department * 100
        X.drop('department', axis=1, inplace=True)
        return X.fillna(0)


class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.column_name, axis=1)


if __name__ == '__main__':
    data_train_ranker = pd.read_csv('../Data/data_train_ranker.csv')
    item_features = pd.read_csv('../Data/item_features.csv')
    user_features = pd.read_csv('../Data/user_features.csv')

    # new feature for user
    additional_user_features = make_pipeline(
        AddPriceColumn(),
        AverageCheck(),
        AveragePriceCounItemADepartment(items_department=item_features[['item_id', 'department']]),
        PurchasesAMonth(),
        DropColumn(column_name='price'),
        UpdateFeaturesTable(feature_table=user_features,
                            X_old_columns_names=data_train_ranker.columns,
                            merge_on=['user_id'])
    )

    user_features = additional_user_features.fit_transform(data_train_ranker)

    additional_items_features = make_pipeline(
        AddPriceColumn(),
        PurchasesCountAWeek(),
        AveragePurchasesCountADepartment(item_department_table=item_features[['item_id', 'department']]),
        LikePercentADepartment(),
        HigherThenMedianPricePercent(),
        UpdateFeaturesTable(feature_table=item_features,
                            X_old_columns_names=data_train_ranker.columns,
                            merge_on=['item_id'])
    )

    item_features = additional_items_features.fit_transform(X=data_train_ranker)
    """
    to = to.merge(item_features, on='item_id', how='left')
    to = to.merge(user_features, on='user_id', how='left')
    """
    print()
