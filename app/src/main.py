import redis.asyncio as redis
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from model import Model
from enum import Enum
import pandas as pd
from fastapi.responses import FileResponse

class Gender(Enum):
    MALE = 'male'
    FEMALE = 'female'

class Education(Enum):
    MIDDLE_SPECIAL='MIDDLE_SPECIAL'
    MIDDLE='MIDDLE'
    HIGHER='HIGHER'
    HIGHER_HALF='HIGHER_HALF'
    MIDDLE_HALF='MIDDLE_HALF'
    SEVERAL_HIGH='SEVERAL_HIGH'
    PHD='PHD'


class MaritalStatus(Enum):
    MARRIED='MARRIED'
    NOT_MARRIED='NOT_MARRIED'
    DIVORCED='DIVORCED'
    WIDOW='WIDOW'
    CIVIL='CIVIL'

education_map = {
    'MIDDLE_SPECIAL': 'Среднее специальное',
    'MIDDLE': 'Среднее',
    'HIGHER': 'Высшее',
    'HIGHER_HALF': 'Неоконченное высшее',
    'MIDDLE_HALF': 'Неполное среднее',
    'SEVERAL_HIGH': 'Два и более высших образования',
    'PHD': 'Ученая степень'
}

marital_status_map = {
    'MARRIED': 'Состою в браке',
    'NOT_MARRIED': 'Не состоял в браке',
    'DIVORCED': 'Разведен(а)',
    'WIDOW': 'Вдовец/Вдова',
    'CIVIL': 'Гражданский брак'
}

r = redis.from_url("redis://redis")

model = Model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info('Starting up...')
    await r.ping()
    model.train_model()
    yield
    # Shutdown
    logging.info('Shutting down...')

app = FastAPI(lifespan=lifespan)

async def get_cached_data(key: bytes):
    cached_data = await r.get(key)
    if cached_data:
        return cached_data
    return None

async def set_cached_data(key, data):
    await r.set(key, data, ex=5)


@app.get("/ping")
async def ping():
    return ''

@app.get("/predict")
async def predict(age: int, gender: Gender, education: Education,
                  marital_status: MaritalStatus):
    key = (str(age) + gender.value).encode()
    cached_predict = await get_cached_data(key)
    if cached_predict:
        return cached_predict
    
    gender = 1 if gender == Gender.MALE else 0
    user_input_df = pd.DataFrame({"AGE": [age],
                                  "GENDER": [gender],
                                  "MARITAL_STATUS": [marital_status_map[marital_status.value]],
                                  "EDUCATION": [education_map[education.value]]})

    full_X_df = pd.concat((user_input_df, model.df), axis=0)
    preprocessed_X_df = model.preprocess_data(full_X_df)

    user_X_df = preprocessed_X_df[:1]

    prediction, prediction_probas = model.predict(user_X_df)
    
    #await set_cached_data(param, b'CACHED!!!!')
    return {"prediction": prediction.item(),
            "prediction_probas": prediction_probas.tolist()}

@app.get("/retrain")
async def retrain_model(param: str):
    pass

@app.get("/weights")
async def get_weights():
    model.save_model()
    return FileResponse("data/model_weights.mw")

@app.get("/accuracy")
async def get_accuracy():
    return model.get_accuracy()