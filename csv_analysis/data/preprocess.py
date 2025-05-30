import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from sklearn import preprocessing
import datetime
import dateutil.parser
VIZ_ROOT = 'Plots'
NUNIQUE_THRESHOLD = 20

class CSVPreProcess:

    def __init__(self, input, target_col = None, index_column = None, exclude_columns = []):
        if type(input)==str:
            self.df = pd.read_csv(input, index_col = index_column)
        else:
            self.df = input
        self.df.drop(exclude_columns,inplace=True)
        self.col_names = list(self.df.columns)
        self.target_column = self.col_names[-1] if target_col == None else target_col
        self.df.dropna(subset=[self.target_column], inplace=True)
        self.num_cols = len(self.col_names)
        self.output_format = 'png'
        self.categorical_data_types = ['object','str']
        self.categorical_column_list = []
        self.populate_categorical_column_list()
        self.numerical_column_list = list(self.get_filtered_dataframe(include_type=np.number))
        temp_col_list = [num_col for num_col in self.numerical_column_list if self.df[num_col].nunique() < NUNIQUE_THRESHOLD]
        self.continuous_column_list = [x for x in self.numerical_column_list if x not in temp_col_list]
        self.non_continuous_col_list = self.categorical_column_list + temp_col_list
        self.converted_date = ''

    def get_filtered_dataframe(self, include_type = [], exclude_type = []):
        if include_type or exclude_type:
            return self.df.select_dtypes(include = include_type, exclude = exclude_type)
        else:
            return self.df

    def populate_categorical_column_list(self):
        df = self.get_filtered_dataframe(exclude_type=np.number)
        if not self.categorical_column_list:
            for column in df:
                if df[column].nunique() <= NUNIQUE_THRESHOLD:
                    self.categorical_column_list.append(column)

    def encode_categorical_target(self):
        if self.target_column in self.categorical_column_list:
            le = preprocessing.LabelEncoder()
            y = self.df[self.target_column]
            le.fit(y)
            self.df[self.target_column] = le.transform(y)

    def fill_numerical_na(self, ret = False):
        columns = self.df.columns[self.df.isna().any()].tolist()
        for col in self.continuous_column_list:
            try:
                if col in columns:
                    x = y = self.df[col]
                    x = x.fillna(self.df[col].mean())
                    mean_corr = x.corr(self.df[self.target_column])
                    y = y.fillna(self.df[col].median())
                    median_corr = y.corr(self.df[self.target_column])
                    if(abs(mean_corr) > abs(median_corr)):
                        self.df[col] = x
                    else:
                        self.df[col] = y
            except Exception as e:
                pass
        if ret:
            return self.df

    def fill_categorical_na(self, ret = False):
        self.df = self.df.fillna("Unknown")
        if ret:
            return self.df

    def normalize_numerical(self):
        for col in self.numerical_column_list:
            if col != self.target_column:
                self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())

    def standardize(self):
        for col in self.df.columns:
            self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()

    def mean_normalization(self):
        for col in self.df.columns:
            self.df[col] = (self.df[col] - self.df[col].mean()) / (self.df[col].max() - self.df[col].min())

    def encode_categorical(self):
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.df.reset_index(drop=True, inplace=True)
        for col in self.categorical_column_list:
            if col != self.target_column:
                enc_df = pd.DataFrame(enc.fit_transform(self.df[[col]]))
                enc_df.columns = enc.get_feature_names_out([col])
                self.df.drop(columns=[col], inplace=True)
                self.df = pd.concat([self.df, enc_df], axis=1)
        self.df.reset_index(drop=True, inplace=True)

    def remove_outliers(self, ret = False):
        df = self.df[self.continuous_column_list]
        del_list = self.continuous_column_list
        self.df = self.df.drop(columns = del_list)
        z_scores = stats.zscore(df)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        df = df[filtered_entries]
        self.df = pd.concat([self.df, df], axis=1, join='inner')
        if ret:
            return self.df

    def remove_non_contributing_features(self):
        cor = self.df.corr()
        del_list = []
        for col1 in self.continuous_column_list:
            for col2 in self.continuous_column_list:
                if col1 != self.target_column and col2 != self.target_column and col1 != col2:
                    cor12 = abs(cor[col1][col2])
                    if cor12 > 0.85:
                        if cor[col1][self.target_column] > cor[col2][self.target_column]:
                            del_list.append(col2)
                        else:
                            del_list.append(col1)
        self.df = self.df.drop(columns = list(set(del_list)), axis = 1)

        del_list = []
        for col in self.continuous_column_list:
            if abs(cor[col][self.target_column]) < 0.1 and col in self.df.columns:
                del_list.append(col)
        self.df = self.df.drop(columns = del_list, axis = 1)
        self.df.to_csv('Preprocess_file.csv')

    def convert_date_format(self, input_date, output_date_format = 'DD/MM/YYYY'):
        output_date_formats = {
            'DD/MM/YYYY': '%d/%m/%Y', 'YYYY/DD/MM': '%Y/%d/%m', 'MM/DD/YYYY': '%m/%d/%Y',
            'YYYY/MM/DD': '%Y/%m/%d', 'DD-MM-YYYY': '%d-%m-%Y', 'YYYY-DD-MM': '%Y-%d-%m',
            'MM-DD-YYYY': '%m-%d-%Y', 'YYYY-MM-DD': '%Y-%m-%d'
        }
        parsed_date = dateutil.parser.parse(input_date, dayfirst=True)
        self.converted_date = parsed_date.strftime(output_date_formats[output_date_format])
        return self.converted_date
