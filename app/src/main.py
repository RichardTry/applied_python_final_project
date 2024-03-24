import redis.asyncio as redis
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

r = redis.from_url("redis://redis")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print('Starting up...')
    await r.ping()
    yield
    # Shutdown
    logging.info('Shutting down...')

app = FastAPI(lifespan=lifespan)

async def get_cached_data(key):
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
async def cached_endpoint(param: str):
    cached_data = await get_cached_data(param)
    if cached_data:
        return {"cached_data": cached_data}
    
    data = {"prediction": "test_prediction"}
    
    await set_cached_data(param, b'CACHED!!!!')
    return {"data": data}
