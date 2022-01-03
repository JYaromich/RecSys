import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from IPython.core.debugger import set_trace
from tqdm import tqdm
from pathlib import Path

from recommenders import BaseRecommender, ClassificationModel, PopularItemsRecommendation, ALSRecommendation, LightGBMRecommendation
from utils import send_tg_message, prefilter_items
from metrics import calc_precision

cat_features = [
    'department',
    'brand',
    'commodity_desc',
    'sub_commodity_desc',
    'curr_size_of_product',
    'age_desc',
    'marital_status_code',
    'income_desc',
    'homeowner_desc',
    'hh_comp_desc',
    'household_size_desc',
    'kid_category_desc',
    'is_department_miss',
    'is_brand_miss',
    'is_commodity_desc_miss',
    'is_sub_commodity_desc_miss',
    'is_curr_size_of_product_miss',
    'is_age_desc_miss',
    'is_marital_status_code_miss',
    'is_income_desc_miss',
    'is_homeowner_desc_miss',
    'is_hh_comp_desc_miss',
    'is_household_size_desc_miss',
    'is_kid_category_desc_miss'
]

class GeneralModel:
    def __init__(self, 
        first_level_model: BaseRecommender, 
        second_level_model: ClassificationModel,
        item_features: pd.DataFrame,
        user_features: pd.DataFrame,
        count_of_first_model_predict=500,
    ) -> None:
        
        self.first_level_model = first_level_model
        self.second_level_model = second_level_model
        self.warm_start = True
        self.count_of_first_model_predict = count_of_first_model_predict
        self.user_col = 'user_id'
        self.popular_recommendation = None
        self.item_features = item_features
        self.user_features = user_features
    
    def fit(self, data):
        self.popular_recommendation = PopularItemsRecommendation(warm_start=True)
        self.popular_recommendation.fit(data)
    
        self.first_level_model.fit()
            
        first_predict = self.__first_model_predict(users=self.get_unique_users(data))
        
        for_second_model_train_data = self.second_level_model.create_user_item_matrix(first_predict, data)
        for_second_model_train_data = self.second_level_model.additional_features(
            to=for_second_model_train_data,
            item_features=self.item_features,
            user_features=self.user_features,
            data_train=data,
        )
        
        for_second_model_train_data = self.__fill_nan(for_second_model_train_data)
        
        
        X_train = for_second_model_train_data.drop('target', axis=1)
        y_train = for_second_model_train_data[['target']]
        X_train = self.reduce_mem_usage(X_train)
        
       
        self.second_level_model.fit(X_train=X_train, y_train=y_train.values.ravel())
        
        send_tg_message('General model is fitted. Everythings is OK')
        
        
    
    def get_unique_users(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            data=data[self.user_col].unique(),
            columns=[self.user_col]
        )
    
    def __first_model_predict(self, users: pd.DataFrame) -> pd.DataFrame:
        users['candidates'] = users[self.user_col].apply(
            lambda x: self.first_level_model.recommend(
                x, 
                N=self.count_of_first_model_predict
            )
        )
        return users
    
    @staticmethod
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

    
    def __fill_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        num_columns_name = data.dtypes[~(data.dtypes == 'object')].index.to_list()
        
        nun_column_name = data.isna().sum()[data.isna().sum() > 0].index.to_list()
        cat_columns = data[nun_column_name].dtypes[data[nun_column_name].dtypes == 'object'].index.to_list()
        num_columns = list(set(nun_column_name) - set(cat_columns))

        for num_column in num_columns:
            mean = round(data[num_column].median(), 3)
            data[num_column] = data[num_column].fillna(mean)

        cat_imputer = SimpleImputer(strategy='most_frequent', add_indicator=True)

        def fill_nan(df_ranker_train_copy, imputer, nan_columns):

            columns = df_ranker_train_copy.columns.to_list()
            columns.extend(['is_' + column + '_miss' for column in nan_columns])
            

            df_ranker_train_copy = pd.DataFrame(
                data=imputer.fit_transform(df_ranker_train_copy),
                columns=columns
            )
            return df_ranker_train_copy

        data = fill_nan(data, cat_imputer, cat_columns)
        
        # преобразование типов
        
        for num_col in num_columns_name:
            data[num_col] = pd.to_numeric(data[num_col])
            
        return data
        
        
    @staticmethod
    def remake_cold_start_to_warm_start(
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame
    ):
                
        common_users = list(
            set(train_data.user_id.values) & (set(test_data.user_id.values))
        )
        
        train_data = train_data[train_data.user_id.isin(common_users)]
        test_data = test_data[test_data.user_id.isin(common_users)]

        return train_data, test_data
    
    def recommned(self, users: pd.DataFrame, N=5):
        tqdm.pandas()
        
        unique_users = self.get_unique_users(users)
        unique_users['recommendation'] = unique_users[self.user_col].progress_apply(
            lambda x: self.postfilter_items(
                user=x,
                recommendations=self.second_level_model.recommend(x, N=N * 100),
                N=N
            )
        )
        return unique_users
    
    def postfilter_items(self, user, recommendations, N=5):
        """Пост-фильтрация товаров
        
        Input
        -----
        recommendations: list
            Ранжированный список item_id для рекомендаций
        item_info: pd.DataFrame
            Датафрейм с информацией о товарах
        """
        
        # Уникальность
        # recommendations = list(set(recommendations)) - неверно! так теряется порядок
        unique_recommendations = []
        [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]
        
        # Разные категории
        # categories_used = []
        final_recommendations = []
        
        # CATEGORY_NAME = 'sub_commodity_desc'
        # for item in unique_recommendations:
        #     category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]
            
        #     if category not in categories_used:
        #         final_recommendations.append(item)
                
        #     unique_recommendations.remove(item)
        #     categories_used.append(category)
        
        # # Для каждого юзера 5 рекомендаций (иногда модели могут возвращать < 5)
        # set_trace()
        # n_rec = len(final_recommendations)
        # if n_rec < N:
        #     # Более корректно их нужно дополнить топом популярных (например)
        #     final_recommendations.extend(self.popular_recommendation.recommend( 
        #         N=np.abs(len(unique_recommendations) - N) 
        #     ))  # (!) это не совсем верно
        # else:
        #     final_recommendations = final_recommendations[:N]
        
        # ! рассширяем по топ полулярными
        if len(unique_recommendations) < N:
            unique_recommendations.extend(self.popular_recommendation.recommend( 
                N=np.abs(len(unique_recommendations) - N) 
            ))
            final_recommendations = unique_recommendations
        else:
            final_recommendations = unique_recommendations[:N]
            
            
        # 2 новых товара (юзер никогда не покупал)
        # your_code
        
        # 1 дорогой товар, > 7 долларов
        # your_code
        
        # assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
        assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
        return final_recommendations
        
        

if __name__ == '__main__':

    DATA_PATH = Path('./Data/')
    RETAIL_TRAIN_PATH = DATA_PATH / 'retail_train.csv'
    ITEM_FEATURE_PATH = DATA_PATH / 'product.csv'
    USER_FEATURE_PATH = DATA_PATH / 'hh_demographic.csv'
    TEST_DATA = DATA_PATH / 'retail_test1.csv'

    train_data = pd.read_csv(RETAIL_TRAIN_PATH)
    item_features = pd.read_csv(ITEM_FEATURE_PATH)
    user_features = pd.read_csv(USER_FEATURE_PATH)
    test_data = pd.read_csv(TEST_DATA)

    ITEM_COL = 'item_id'
    USER_COL = 'user_id'

    # column processing
    item_features.columns = [col.lower() for col in item_features.columns]
    user_features.columns = [col.lower() for col in user_features.columns]

    item_features.rename(columns={'product_id': ITEM_COL}, inplace=True)
    user_features.rename(columns={'household_key': USER_COL }, inplace=True)


    train_data, test_data = GeneralModel.remake_cold_start_to_warm_start(train_data, test_data)

    best_params_for_second_models = {
        'learning_rate': 0.08, 
        'max_depth': 40, 
        # 'min_child_samples': 3, 
        'min_data_in_leaf': 180, 
        'n_estimators': 30, 
        'num_leaves': 2
    }
    input_data = prefilter_items(
                    data=train_data,
                    items_data=item_features,
                    take_n_popular=5000,
                    warm_start=True
                )

    general_model_params = {
        'first_level_model': ALSRecommendation(
            data=input_data,
            item_features=item_features,
            iterations=70,
            n_factors=400,
            regularization=0.018,
            warm_start=True
        ),
        
        'second_level_model': LightGBMRecommendation(
            categorical_column=cat_features,
            **best_params_for_second_models
        ),
        'item_features': item_features,
        'user_features': user_features
    }  

    general_model = GeneralModel(**general_model_params)
    general_model.fit(input_data)

    RECOMEND_COUNT = 5

    # ! Test data
    data = test_data

    result = data.groupby('user_id')['item_id'].unique().reset_index()
    result.columns=['user_id', 'actual']

    result = result.merge(
        general_model.recommned(
            users=general_model.get_unique_users(data),
            N=RECOMEND_COUNT
        ),
        on='user_id',
        how='left'
    )

    TOPK_PRECISION=5

    print(sorted(calc_precision(result, TOPK_PRECISION), key=lambda x: x[1], reverse=True))



