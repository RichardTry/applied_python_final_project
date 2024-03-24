from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd
import numpy as np

class Model:
    def __init__(self):
        pass

    def load_data(self):
        data = dict()
        tables = ('clients',
                'close_loan',
                'job',
                'last_credit',
                'loan',
                'pens',
                'salary',
                'target',
                'work')
        for table in tables:
            data[table] = pd.read_csv('data/D_' + table + '.csv')
        return data

    def join_data(self, data):
        df = data['target']
        df = df.merge(data['clients'],
                left_on='ID_CLIENT',
                right_on='ID', how='left')
        df = df.drop(columns=['ID'])
        df = df.merge(data['salary'],
                left_on='ID_CLIENT',
                right_on='ID_CLIENT', how='left')
        df = df.merge(
            data['loan'].groupby(['ID_CLIENT']).agg(LOAN_NUM_TOTAL=('ID_LOAN',
                                                                    'count')),
            left_on='ID_CLIENT',
            right_on='ID_CLIENT', how='left')
        closed_per_client = data['loan'].merge(
            data['close_loan'],
            left_on='ID_LOAN',
            right_on='ID_LOAN', how='left').groupby(['ID_CLIENT']).agg(LOAN_NUM_CLOSED=('CLOSED_FL',
                                                                    'count'))
        df = df.merge(
            closed_per_client,
            left_on='ID_CLIENT',
            right_on='ID_CLIENT', how='left')
        df = df.drop(columns=['ID_CLIENT',
                            'REG_ADDRESS_PROVINCE',
                            'POSTAL_ADDRESS_PROVINCE',
                            'FACT_ADDRESS_PROVINCE'])
        return df

    def open_data(self):
        self.df = self.join_data(self.load_data())

    def preprocess_data(self, df: pd.DataFrame, target_included=False):

        if target_included:
            X_df, y_df = self.drop_target(df)
        else:
            X_df = df
        
        features_whitelist = ['AGE', 'GENDER', 'MARITAL_STATUS', 'EDUCATION']
        X_df = X_df.loc[:, features_whitelist]

        to_encode = ['EDUCATION', 'MARITAL_STATUS', 'GENDER']
        for col in to_encode:
            dummy = pd.get_dummies(X_df[col], prefix=col)
            X_df = pd.concat([X_df, dummy], axis=1)
            X_df.drop(col, axis=1, inplace=True)

        X_df.dropna(inplace=True)

        if target_included:
            return X_df, y_df
        else:
            return X_df

    def drop_target(self, df: pd.DataFrame):
        y = df['TARGET']
        X = df.drop(columns=['TARGET'])

        return X, y
    
    def train_model(self, df=None):
        if df is None:
            self.open_data()
        else:
            df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != 'O' else x.fillna(x.mode()[0]), axis=0)
            self.df = df
        X_df, y_df = self.preprocess_data(self.df, target_included=True)
        self.fit(X_df, y_df)

    def fit(self, X_df, y_df):
        self.model = RandomForestClassifier()
        self.model.fit(X_df, y_df)

        test_prediction = self.model.predict(X_df)
        self.accuracy = accuracy_score(test_prediction, y_df)

    def save_model(self, path="data/model_weights.mw"):
        with open(path, "wb") as file:
            dump(self.model, file)

    def get_accuracy(self):
        return self.accuracy

    def load_model(self, path="data/model_weights.mw"):
        with open(path, "rb") as file:
            return load(file)


    def predict(self, df):
        prediction = self.model.predict(df)[0]
        prediction_proba = self.model.predict_proba(df)
        prediction_proba = np.squeeze(prediction_proba)

        return prediction, prediction_proba
