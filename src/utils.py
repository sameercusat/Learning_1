import dill
import sys
import os
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path ,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)     

def evaluate_model(x_train,y_train,x_test,y_test,models):
    reports={}
    logging.info("Evaluating Models")
    for i in range(len(list(models.keys()))):
        model=list(models.values())[i]
        model.fit(x_train,y_train)
        y_train_pred=model.predict(x_train)
        y_test_pred=model.predict(x_test)
        r2_score_train=r2_score(y_train,y_train_pred)
        r2_score_test=r2_score(y_test,y_test_pred)
        reports[list(models.keys())[i]]=r2_score_test
    logging.info("Model Evaluation Ends")    
    return reports
