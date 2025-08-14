from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from src.utils import evaluate_model
import sys
import os

@dataclass
class Model_Trainer_Config:
    trained_model_file=os.path.join("artifacts","model_trainer.pkl")

class ModelTrainer:

    def __init__(self):
        self.modelTrainerConfig=Model_Trainer_Config()

    def modelTrainer(self,train_array,test_array):
        try:
                logging.info("Splitting data into 4 part x_train,y_train,x_test,y_test")
                x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
                logging.info("Sending Data to models")

                models={'Linear_Regression':LinearRegression(),
                        'Ridge':Ridge(),
                        'Lasso':Lasso(),
                        'Random_Forest':RandomForestRegressor(),
                        'Decsion_Tree':DecisionTreeRegressor(),
                        'Support_Vector':SVR(),
                        'Kneighbours':KNeighborsRegressor()}
                
                reports:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
                best_model_name,best_model_score=max(reports.items(),key=lambda x:x[1])
                best_model=models[best_model_name]

                if best_model_score <0.6:
                    raise CustomException("No Best Model Found")
                

                logging.info(f"Best Model is {best_model_name}")

                save_object(
                    file_path=self.modelTrainerConfig.trained_model_file,
                    obj=best_model
                )
                logging.info("Model saved into Pickle File")

                y_test_pred=best_model.predict(x_test)

                best_model_r2_score=r2_score(y_test,y_test_pred)

                return best_model_r2_score

        except Exception as e:
             raise CustomException(e,sys)

