import dill
import sys
import os
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging
from sklearn.model_selection import GridSearchCV
import pickle

def save_object(file_path,obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path ,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)     

def load_object(file_path_obj):
    try:
        with open(file_path_obj,"rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    reports={}
    logging.info("Evaluating Models")
    for i in range(len(list(models.keys()))):
        param_grid=param[list(models.keys())[i]]
        model=list(models.values())[i]
        gs=GridSearchCV(estimator=model,param_grid=param_grid,cv=3)
        gs.fit(x_train,y_train)
        model.set_params(**gs.best_params_)
        model.fit(x_train,y_train)
        y_train_pred=model.predict(x_train)
        y_test_pred=model.predict(x_test)
        r2_score_train=r2_score(y_train,y_train_pred)
        r2_score_test=r2_score(y_test,y_test_pred)
        logging.info(f'The model is {list(models.keys())[i]} ,parameters are {gs.best_params_} and r2_score is {r2_score_test}')
        reports[list(models.keys())[i]]=r2_score_test
    logging.info("Model Evaluation Ends")    
    return reports
