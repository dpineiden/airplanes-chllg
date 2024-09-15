import fastapi
from pydantic import BaseModel
from .model import DelayModel
from .enumerations import (
    OPERADORAS, TIPOVUELOS, SIGLADES, DIANOM,
    EMPRESA, DESTINATION, OPNMR, FLNMR)
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
import random
from fastapi import FastAPI, HTTPException

app = fastapi.FastAPI()

delay_model = DelayModel()
delay_model.load_model()  # Load the saved model

import calendar

# Get a dictionary of months and their maximum number of days


@dataclass
class Flight:
    opera: str
    type: str
    day: int
    month: int
    year: int

    def item(self):
        hour = random.choice(range(0, 24))
        min = random.choice(range(0, 60))
        delta = random.choice(range(0,30))
        day = datetime(
            self.year, 
            self.month, 
            self.day, hour, min)
        w = day.weekday()
        return {
            "OPERA":self.opera,
            "TIPOVUELO":self.type,
            "DIA":self.day,
            "MES":self.month,
            "AÑO":self.year,
            "Fecha-I": str(day),
            "Fecha-O": str(day+timedelta(hours=delta))   ,
            "Vlo-I":str(random.choice(FLNMR))     ,
            "Ori-I":"SCEL",
            "Vlo-O":str(random.choice(OPNMR))    ,
            "Ori-O":"SCEL",
            "Des-O":random.choice(DESTINATION) ,
            "Emp-O":random.choice(EMPRESA)     ,
            "Des-I":random.choice(DESTINATION)     ,
            "Emp-I":random.choice(EMPRESA)     ,
            "DIANOM":DIANOM[w],
            "SIGLAORI": "Santiago"  ,
            "SIGLADES": random.choice(SIGLADES)  ,

        }


def create_flights(data:dict[str,Any])->List[Flight]:
    day = data.get("DIA")
    month = data.get("MES")
    if not (1<=month<13):
        raise Exception("Month value incorrect")
    year = data.get("AÑO")
    opera = data.get("OPERA")
    ftype = data.get("TIPOVUELO")
    if opera in OPERADORAS  and ftype in TIPOVUELOS:
        if not year:
            year = datetime.now().year
        max_days_by_month = {
            month: calendar.monthrange(year, month)[1] for month in range(1, 13)}

        months = []
        if not month:
            months = [i+1 for i in range(13)]
        else:
            months= [month]
        days = []
        if not day:
            days = {month:[i+1 for i in range(max_days_by_month[month])]
                    for month in months}
        records = []
        for month  in months:
            month_days = days[month]
            for day in month_days:
                item = {
                    "opera":opera,
                    "type":ftype,
                    "year":year,
                    "month":month,
                    "day":day
                }
                fl = Flight(**item)
                records.append(fl.item())
        return records
    else:
        raise Exception("TIPOVUELO given not exists, or OPERADORA not exists")


class PredictionInput(BaseModel):
    flights: List[dict]  # Accepting a list of dictionaries for the input data

    def create(self):
        records = []
        for flight in self.flights:
            flights = create_flights(flight)
            records.extend(flights)
        return records 

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(input_data:PredictionInput) -> dict:
    try:
            # Make predictions
        flights = input_data.create()
        data = pd.DataFrame(flights)
        # Preprocess the input data
        preprocessed_data, target = delay_model.preprocess(data,"delay")


        predictions = delay_model.predict(preprocessed_data)

        # Return the predictions as a response
        return {
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed: {str(e)}")
