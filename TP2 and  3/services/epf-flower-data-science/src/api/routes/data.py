from fastapi import APIRouter
from src.schemas.message import MessageResponse
from src.services import data as data_func
router = APIRouter()


@router.get("/data/{data}", name="Demo route", response_model=MessageResponse)
def data(data: str) -> MessageResponse:
    data_func.download_dataset(data)
    return MessageResponse(message=f"Data downloaded")

@router.post("/add_dataset", name="Add Dataset to JSON")
def add_dataset(name: str, url: str)-> MessageResponse:    
    data_func.add_dataset(name, url)
    return MessageResponse(message = f"Data added !")

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




    



