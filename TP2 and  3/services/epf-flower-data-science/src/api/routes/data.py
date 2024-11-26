from fastapi import APIRouter
from src.schemas.message import MessageResponse
from src.services import data as data_func
router = APIRouter()


@router.get("/data/{data}", name="Demo route", response_model=MessageResponse)
def data(data: str) -> MessageResponse:
    data_func.download_dataset(data)
    # return MessageResponse(message=f"Here is your dataset : {dataset}")



