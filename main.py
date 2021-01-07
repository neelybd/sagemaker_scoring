from mlio.integ.numpy import as_numpy
import mlio # ver 0.7
import sklearn, pickle # scikit-learn v 0.23.2
import numpy as np
import xgboost as xgb
from artifacts.dataprocessing.code.sagemaker_serve import model_fn, output_fn, predict_fn, read_csv_data
import artifacts.dataprocessing.code.dpp0 as dpp
import os
import pandas as pd
from file_handling_docker import *


def recordio_protobuf_to_dmatrix(string_like):  # type: (bytes) -> xgb.DMatrix
    """Convert a RecordIO-Protobuf byte representation to a DMatrix object.
    Args:
        string_like (bytes): RecordIO-Protobuf bytes.
    Returns:
    (xgb.DMatrix): XGBoost DataMatrix
    """
    buf = bytes(string_like)
    dataset = [mlio.InMemoryStore(buf)]
    reader_params = mlio.DataReaderParams(dataset=dataset, batch_size=100)
    reader = mlio.RecordIOProtobufReader(reader_params)

    is_dense_tensor = type(reader.peek_example()['values']) is mlio.DenseTensor

    examples = []
    for example in reader:
        # Ignore labels if present
        values = as_numpy(example['values']) if is_dense_tensor else to_coo_matrix(example['values'])
        examples.append(values)

    data = np.vstack(examples) if is_dense_tensor else scipy_vstack(examples).tocsr()
    dmatrix = xgb.DMatrix(data)
    return dmatrix


# Import Data Location
data_in_path = os.path.join("data", "forecast.csv")

# Model Location
model_path = os.path.join("artifacts", "tuning", "xgboost-model")
model_fn_path = os.path.join("artifacts", "dataprocessing")

# Import Data
data = open_unknown_csv(data_in_path, ',')

# Get X and y
X = data.drop(columns=[dpp.HEADER.target_column_name])
y = data[dpp.HEADER.target_column_name]
# X, y = read_csv_data(source=data_in_path,
#                      target_column_index=dpp.HEADER.target_column_index,
#                      output_dtype='O')

# Import model
model = pickle.load(open(model_path, "rb"))
clf1 = model_fn(model_fn_path)

# Score data
data_scoring = output_fn((predict_fn(X, clf1), None), "application/x-recordio-protobuf")
rec = recordio_protobuf_to_dmatrix(data_scoring.data)
data_scored = (model.predict(rec))

# Merge
data["Predicted"] = data_scored

data.to_csv("out.csv")
