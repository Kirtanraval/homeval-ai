from pydantic import BaseModel

class HouseData(BaseModel):

    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int