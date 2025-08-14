import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_trasformation import DataTransformer
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train_data.csv')
    test_data_path:str=os.path.join('artifacts','test_data.csv')
    raw_data_path:str=os.path.join("artifacts",'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.dataIngestionConfig=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Begins")
        try:
            logging.info("Converting .csv into Dataframe")
            df=pd.read_csv(r"notebook\stud.csv")
            logging.info("Making artifacts directory if it does not exist")
            os.makedirs(os.path.dirname(self.dataIngestionConfig.train_data_path),exist_ok=True)
            df.to_csv(self.dataIngestionConfig.raw_data_path,index=False,header=True)
            logging.info("Train Test Split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.dataIngestionConfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.dataIngestionConfig.test_data_path,index=False,header=True)
            logging.info("Data Ingestion Completed")
            return (self.dataIngestionConfig.train_data_path,self.dataIngestionConfig.test_data_path)
        
        except Exception as e:
            logging.info(f"Error Occured while Data Ingestion {e}")
            raise CustomException(e,sys)


if __name__ =="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    tranformation_obj=DataTransformer()
    trainer_obj=ModelTrainer()
    #print("train_path: ",train_path)
    #print("test_data: ",test_path)
    x,y,_=tranformation_obj.initiate_data_transformation(train_path,test_path)
    r2_score=trainer_obj.modelTrainer(x,y)
    print(r2_score)





