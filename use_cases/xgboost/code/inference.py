import multiprocessing
import os
import traceback
import json
import xgboost as xgb
import pandas as pd
from typing import List

import whylogs as why
from whylogs.api.writer import Writer, Writers
from whylogs.api.logger.experimental.logger.actor.thread_rolling_logger import ThreadRollingLogger
from whylogs.api.logger.experimental.logger.actor.time_util import Schedule, TimeGranularity

# Initialize whylogs with your WhyLabs API key and target dataset ID. You can get an api key from the
# settings menu of you WhyLabs account.
why.init() # This loads credentials from the env directly

def create_logger():
    logger = ThreadRollingLogger(
        # This should match the model type in WhyLabs. We're using a daily model here.
        aggregate_by=TimeGranularity.Day,
        # The profiles will be uploaded from the rolling logger to WhyLabs every 5 minutes. Data
        # will accumulates during that time.
        write_schedule=Schedule(cadence=TimeGranularity.Minute, interval=5),
        writers=[Writers.get('whylabs')]
    )

    return logger

logger = create_logger()

def model_fn(model_dir):
    model_file = "xgboost-model"
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, model_file))
    return booster


def input_fn(request_body, request_content_type):
    print(f'Logger closed: {logger.is_closed()}, alive : {logger.is_alive()}')

    assert request_content_type == 'application/json'
    # Body should be a list of lists of length 4
    body = json.loads(request_body)

    if 'flush' in body and body['flush']:
        # Utility for flushing the logger, which forces it to upload any pending profiles synchronously.
        print("Flushing logger...")
        logger.flush()
        print("Done flushing logger")
        return None

    if 'close' in body and body['close']:
        logger.close()
        return None

    if type(body) is not list:
        raise ValueError(f"Expected a list of lists, got {type(body)}")

    if len(body) == 0:
        raise ValueError("Expected a list of lists, got an empty list")

    if len(body[0]) != 4:
        raise ValueError(f"Expected a list of lists of length 4, got a list of lists of length {len(body[0])}")

    return body


column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def predict_fn(input_data: List[List[float]], model):
    if input_data is None:
        return ""

    test = xgb.DMatrix(input_data)
    predictions = model.predict(test)

    df = pd.DataFrame(input_data, columns=column_names)
    df['prediction'] = predictions.tolist()

    try:
        print("Logging prediction...")
        logger.log(df)
        print("Done logging prediction")
    except Exception as e:
        print(f"Failed to log prediction: {e}")
        print(traceback.format_exc())

    return predictions.tolist()


def output_fn(prediction, content_type):
    return json.dumps(prediction)

