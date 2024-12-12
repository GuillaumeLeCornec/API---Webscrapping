from fastapi import APIRouter
from src.schemas.message import MessageResponse
from src.services import data as data_func
router = APIRouter()
import json

@router.get("/data/{data}", name="Demo route", response_model=MessageResponse)
def data(data: str) -> MessageResponse:
    """Download a dataset.

    Args:
        data (str): The identifier for the dataset to download.

    Returns:
        MessageResponse: A response object indicating the dataset was downloaded.
    """
    data_func.download_dataset(data)
    return MessageResponse(message=f"Data downloaded")

@router.post("/add_dataset", name="Add Dataset to JSON")
def add_dataset(name: str, url: str): 
    """Add a new dataset to the JSON.

    Args:
        name (str): The name of the dataset.
        url (str): The URL of the dataset.

    Returns:
        MessageResponse: A response object indicating the dataset was added, followed by the dataset itself.
    """   
    data = data_func.add_dataset(name, url)
    return MessageResponse(message = f"Data added ! : {data}")


@router.post("/modif_dataset", name="Modify dataset")
def modifiy_dataset(original_name: str, original_url: str, new_name : str, new_url : str)-> MessageResponse:
    """Modify an existing dataset in the JSON.

    Args:
        original_name (str): The original name of the dataset.
        original_url (str): The original URL of the dataset.
        new_name (str): The new name for the dataset.
        new_url (str): The new URL for the dataset.

    Returns:
        MessageResponse: A response object indicating the dataset was modified.
    """    
    data_func.modify_dataset(original_name, original_url, new_name, new_url)
    return MessageResponse(message = f"Data modified !")

@router.get("/load-iris-dataset")
def get_iris_dataset():
    """
    Charger le dataset Iris en tant que DataFrame, obtenir sa forme, et le retourner en JSON.

    Returns:
        dict: A dictionary containing the dataset's shape and its content 
        as a JSON-compatible list of records.
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
    Load and process the Iris dataset.

    This function processes the Iris dataset by applying transformations or 
    cleaning operations and returns the processed dataset along with its shape.

    Returns:
        dict: A dictionary containing the processed dataset's shape and its content 
        as a JSON-compatible list of records.
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
    Split the Iris dataset into training and testing sets.

    Args:
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        dict: A dictionary containing the training and testing datasets as 
        JSON-compatible lists of records.
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
    """
    Train a model on the Iris dataset.

    This function trains a model using a specified training dataset and 
    model parameters and saves the trained model.

    Returns:
        dict: A message indicating the training status.
    """
    train_test_dataset = split_iris_dataset(0.2)
    
    file_path = "C:/Users/Le Cornec/Desktop/EPF/5A/data_sources2/API---Webscrapping/TP2 and  3/services/epf-flower-data-science/src/config/model_parameters.json"
    
    with open(file_path, "r") as file:
        model_para = json.load(file)
    training_model = model_para["model"]["parameters"]
    message = data_func.train_and_save(train_test_dataset, training_model)
    return {"message": message}

@router.get("/get_pred")
def get_predictions():
    """
    Generate predictions using the trained model.

    This function uses the trained model to make predictions on the test dataset.

    Returns:
        dict: A dictionary containing the predictions.
    """
    train_test_dataset = split_iris_dataset(0.2)
    
    file_path = "C:/Users/Le Cornec/Desktop/EPF/5A/data_sources2/API---Webscrapping/TP2 and  3/services/epf-flower-data-science/src/config/model_parameters.json"
    
    with open(file_path, "r") as file:
        model_para = json.load(file)
    training_model = model_para["model"]["parameters"]
    y_pred = data_func.get_pred(train_test_dataset, training_model)
    y_pred_list = y_pred.tolist()

    y_pred_json = {"predictions": y_pred_list}
    return {"message" : y_pred_json}

@router.get("/get_firestore_parameters")
def get_firestore_parameters(collection_name, document_id):
    """
    Retrieve parameters from a Firestore document.

    Args:
        collection_name (str): The name of the Firestore collection.
        document_id (str): The ID of the Firestore document.

    Returns:
        dict: The parameters stored in the specified document.
    """
    parameters = data_func.get_firestore_params(collection_name, document_id)
    return {"message" : parameters}

@router.post("/add_firestore_parameters")
def add_firestore_parameters(collection_name, document_id, data):
    """
    Add parameters to a Firestore document.

    Args:
        collection_name (str): The name of the Firestore collection.
        document_id (str): The ID of the Firestore document.
        data (dict): The data to add to the document.

    Returns:
        dict: The updated document parameters.
    """
    data_func.add_firestore_params(collection_name, document_id, data)
    parameters = data_func.get_firestore_params(collection_name, document_id)
    return {"message" : parameters}

@router.post("/update_firestore_parameters")
def update_firestore_parameters(collection_name, document_id, data):
    """
    Update parameters in a Firestore document.

    Args:
        collection_name (str): The name of the Firestore collection.
        document_id (str): The ID of the Firestore document.
        data (dict): The data to update in the document.

    Returns:
        dict: The updated document parameters.
    """
    data_func.update_firestore_params(collection_name, document_id, data)
    parameters = data_func.get_firestore_params(collection_name, document_id)
    return {"message" : parameters}





    



    



