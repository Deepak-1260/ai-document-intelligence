from pydantic import BaseModel

class CountryParser(BaseModel):
    name:str 
    year:int