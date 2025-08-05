from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import sys
from src.utils import save_object

@dataclass
class Data_tranformation_config:
    data_tranform_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_transform_config=Data_tranformation_config()
    
    def get_dataTransformer_object(self):
        logging.info("Data Transformation Begins")
        try:
            logging.info("Data Transformation Object Creation is Initiated")
            numerical_columns=['reading_score', 'writing_score']
            logging.info(f'Numerical Features:{numerical_columns}')
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            logging.info(f'Categorical Features:{categorical_columns}')
            numerical_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
            categorical_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder())])
            ct=ColumnTransformer([('numerical',numerical_pipeline,numerical_columns),('categorical',categorical_pipeline,categorical_columns)])
            logging.info("Data Transformation Object Craetion Completed")
            return ct
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        logging.info("Iniated Data Transmission")
        try:
            target_column='math_score'
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            transformer_obj=self.get_dataTransformer_object()

            train_independent_features=train_df.drop([target_column],axis=1)
            print(train_independent_features.columns)
            train_dependent_features=train_df[target_column]

            test_independent_features=test_df.drop([target_column],axis=1)
            print(test_independent_features.columns)
            test_dependent_features=test_df[target_column]

            logging.info("Applying Data Transformation on Independent Features")

            trnasformed_train_independent_features=transformer_obj.fit_transform(train_independent_features)
            trnasformed_test_independent_features=transformer_obj.transform(test_independent_features)

            training_array=np.c_[trnasformed_train_independent_features,np.array(train_dependent_features)]
            test_array=np.c_[trnasformed_test_independent_features,np.array(test_dependent_features)]

            save_object(
                file_path=self.data_transform_config.data_tranform_file_path,
                obj=transformer_obj
            )

            return(training_array,test_array,self.data_transform_config.data_tranform_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)
