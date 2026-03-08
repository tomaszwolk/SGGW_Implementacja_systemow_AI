import numpy as np
import mlflow
import mlflow.sklearn

run_id = "d8161d49d4ac488a866a1324d29046dd"
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

print(loaded_model)

example_input = np.array([[2, 1, 28.0, 0, 0, 13.0, False, True]])
prediction = loaded_model.predict(example_input)
print("Prediction for example input:", prediction)

# Z Model Registry
model_name = "Titanic"
version = 1
model = mlflow.sklearn.load_model(
    f"models:/{model_name}/{version}"
)

print(model)

prediction = model.predict(example_input)
print("Prediction for example input (z rejestru):", prediction)

# Z Model Registry z użyciem aliasu
model_name = "Titanic"
model = mlflow.sklearn.load_model(
    f"models:/{model_name}@production"
)

print(model)

prediction = model.predict(example_input)
print("Prediction for example input (z rejestru z aliasem):", prediction)