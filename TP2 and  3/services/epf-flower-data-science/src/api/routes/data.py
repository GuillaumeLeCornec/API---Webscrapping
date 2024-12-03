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

@router.post("/modif_dataset", name="Add Dataset to JSON")
def modifiy_dataset(original_name: str, original_url: str, new_name : str, new_url : str)-> MessageResponse:    
    data_func.modify_dataset(original_name, original_url, new_name, new_url)
    return MessageResponse(message = f"Data modified !")




