import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

#iris_classifier is the name of the service
#runners need not to be only ML models it can be preprocessor.pkl or scaler.pkl as well. we need to mention the order like preprocessor,scaler,model
#runners=[preprocessor,scaler,model], so that it gets executed in the order mentioned
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

#this code is responsible for creating the API for the service
#classify gets created as POST request in deployed local API website
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    input_series = np.asarray(input_series, dtype=np.float32)
    result = iris_clf_runner.predict.run(input_series)
    return result

# command to run: bentoml serve service.py:svc --reload
# give input as [[5.9, 3., 5.1, 1.8]] in the deployed local API website