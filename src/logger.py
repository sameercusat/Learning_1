import logging
from datetime import datetime
import os

LOG_FILE_NAME=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

log_folder=os.path.join(os.getcwd(),"logs")
os.makedirs(log_folder,exist_ok=True)


LOG_FILE_PATH=os.path.join(log_folder,LOG_FILE_NAME)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


if __name__ =='__main__':
    logging.info("Logging has started")