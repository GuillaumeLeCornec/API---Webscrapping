import uvicorn
import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from src.app import get_application


# Initialisation de l'application FastAPI
app = get_application()

@app.get("/")
def root():
    
    """
    Redirige vers la documentation Swagger.
    """
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
