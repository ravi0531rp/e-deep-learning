import io
import os
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from enum import Enum
import ast
import tensorflow as tf
import gdown
from shutil import move

app = FastAPI(title="HorsesVsHumans")
MODEL_TYPE = os.getenv("MODEL_TYPE")
models = dict()


class ModelName(str, Enum):
    horses_vs_humans = "horses_vs_humans"
    
def read_imagefile(file) -> Image.Image:
    image = np.array(Image.open(io.BytesIO(file)))
    return image


@app.on_event("startup")
async def startup_event():
    logger.info(MODEL_TYPE)
    if MODEL_TYPE == ModelName.horses_vs_humans.value:
        if os.path.isfile("./horses-humans.h5"):
            logger.info("Model already Exists...")
        else:
            logger.info("Downloading Model")
            gdown.download("https://drive.google.com/file/d/1zIfDnuFygnWvo1zPOoTKYStNZQaFjExy/view?usp=sharing", fuzzy=True)

        models["predictor"] = tf.keras.models.load_model("./horses-humans.h5")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "Error": "invalid Request"}),
    )


@app.post("/predict")
async def predict_cas(file: UploadFile = File(...)):
    results = dict()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        logger.error(f"ambiguous image format with extension {extension}")

    else:
        try:
            image = read_imagefile(await file.read())
            results["output"] = models["predictor"].predict(image)
            
            logger.debug(results)
        except Exception as e:
            logger.error(f"{repr(e)}")

    return results


@app.get("/health")
def health_check():
    response = {"status": "200", "message": "OK"}
    return response
