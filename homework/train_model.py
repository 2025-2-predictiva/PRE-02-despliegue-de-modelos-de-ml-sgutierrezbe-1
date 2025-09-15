import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar los datos
df = pd.read_csv("/home/ptro/Code/un/semestre 6/predictiva/PRE-02-despliegue-de-modelos-de-ml-Origimed/files/input/house_data.csv", sep=",")

# Seleccionar las características y el objetivo
features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]
target = df[["price"]]

# Entrenar el modelo
estimator = LinearRegression()
estimator.fit(features, target)



with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump(estimator, file)