import bentoml
import numpy as np

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

#.init_local() is used to initialize the model locally
iris_clf_runner.init_local()
a=[[5.9, 3., 5.1, 1.8]]
print(iris_clf_runner.predict.run(a))
print(type(a))
a=np.array(a, dtype=np.float32)
print(type(a))
print(type(iris_clf_runner.predict.run(a)))