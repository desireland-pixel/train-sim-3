# simulation/data_loader.py
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

def load_csv(name):
    path = DATA_DIR / name
    if path.exists():
        try:
            df = pd.read_csv(path)
            if df.empty or len(df.columns) == 0:
                return pd.DataFrame()
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()

def load_all():
    trains = load_csv("trains.csv")
    warehouses = load_csv("warehouses.csv")
    packages = load_csv("packages.csv")
    persons = load_csv("persons.csv")
    points = load_csv("points.csv")
    return {
        "trains": trains,
        "warehouses": warehouses,
        "packages": packages,
        "persons": persons,
        "points": points
    }
