from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import numpy as np

# Cargar el modelo
model = load("model/pokemon-stats-v1.joblib")

# Instanciar la aplicación de FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:35417"],  # Cambia a ["*"] para permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear un modelo de datos para las entradas
class PokemonStats(BaseModel):
    HP: float
    Attack: float
    Defense: float
    SP_Atk: float
    SP_Def: float
    Speed: float

# Endpoint para el mensaje de bienvenida
@app.post("/predict")
async def predict(data: dict):
    # lógica de predicción
    return {"prediction": "result"}

# Endpoint para realizar la predicción
@app.post("/predict")
def predict(stats: PokemonStats):
    # Convertir los datos de entrada en el formato adecuado
    input_data = np.array([[stats.HP, stats.Attack, stats.Defense, stats.SP_Atk, stats.SP_Def, stats.Speed]])
    # Realizar la predicción
    prediction = model.predict(input_data)
    # Retornar la predicción como un JSON
    return {"Total": prediction[0]}
