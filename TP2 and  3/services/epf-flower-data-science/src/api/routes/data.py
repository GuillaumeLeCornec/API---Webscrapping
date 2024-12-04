from fastapi import APIRouter
from src.schemas.message import MessageResponse
from src.services import data as data_func
router = APIRouter()
import json

@router.get("/data/{data}", name="Demo route", response_model=MessageResponse)
def data(data: str) -> MessageResponse:
    data_func.download_dataset(data)
    return MessageResponse(message=f"Data downloaded")

@router.post("/add_dataset", name="Add Dataset to JSON")
def add_dataset(name: str, url: str):    
    data = data_func.add_dataset(name, url)
    # return MessageResponse(message = f"Data added !")
    return data


@router.post("/modif_dataset", name="Modify dataset")
def modifiy_dataset(original_name: str, original_url: str, new_name : str, new_url : str)-> MessageResponse:    
    data_func.modify_dataset(original_name, original_url, new_name, new_url)
    return MessageResponse(message = f"Data modified !")

# @router.get("/load_dataset", name="Load a dataset")
# def load_dataset(name)-> MessageResponse:    
#     data_func.load_dataset(name)
#     return MessageResponse(message = f"Data loaded !")

@router.get("/load-iris-dataset")
def get_iris_dataset():
    """
    Charger le dataset Iris en tant que DataFrame, obtenir sa forme, et le retourner en JSON.
    """
    iris_df = data_func.load_iris_dataset() 
    
    # Obtenir la forme du DataFrame
    dataset_shape = iris_df.shape
    
    # Convertir le DataFrame en JSON
    dataset_json = iris_df.to_dict(orient="records")
    
    return {"message": dataset_shape, "dataset": dataset_json}

@router.get("/process-iris-dataset")
def process_iris_dataset():
    """
    Charger le dataset Iris en tant que DataFrame, obtenir sa forme, et le retourner en JSON.
    """
    processed_iris_df = data_func.process_iris_dataset() 
    
    # Obtenir la forme du DataFrame
    dataset_shape = processed_iris_df.shape
    
    # Convertir le DataFrame en JSON
    dataset_json = processed_iris_df.to_dict(orient="records")
    
    return {"message": dataset_shape, "dataset": dataset_json}

@router.get("/split-iris-dataset")
def split_iris_dataset(test_size):
    """
    Charger le dataset Iris en tant que DataFrame, obtenir sa forme, et le retourner en JSON.
    """
    train, test = data_func.split_iris_dataset(float(test_size))
    train_json = train.to_dict(orient="records")
    test_json = test.to_dict(orient="records")
    
    splitted_iris_json = {
        "train_set": train_json,
        "test_set": test_json
    }
    return {"dataset": splitted_iris_json}
    #works

@router.get("/train-data")
def train_data():
    train_test_dataset = split_iris_dataset(0.2)
    
    file_path = "C:/Users/Le Cornec/Desktop/EPF/5A/data_sources2/API---Webscrapping/TP2 and  3/services/epf-flower-data-science/src/config/model_parameters.json"
    
    with open(file_path, "r") as file:
        model_para = json.load(file)
    training_model = model_para["model"]["parameters"]
    message = data_func.train_and_save(train_test_dataset, training_model)
    return {"message": message}

@router.get("/get_pred")
def get_predictions():
    train_test_dataset = split_iris_dataset(0.2)
    
    file_path = "C:/Users/Le Cornec/Desktop/EPF/5A/data_sources2/API---Webscrapping/TP2 and  3/services/epf-flower-data-science/src/config/model_parameters.json"
    
    with open(file_path, "r") as file:
        model_para = json.load(file)
    training_model = model_para["model"]["parameters"]
    y_pred = data_func.get_pred(train_test_dataset, training_model)
    y_pred_list = y_pred.tolist()

    y_pred_json = {"predictions": y_pred_list}
    return {"message" : y_pred_json}


    



    



