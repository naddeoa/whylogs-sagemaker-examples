import traceback
import torch
import requests
import json
import os

import whylogs as why
from whylogs.api.logger.experimental.logger.actor.process_rolling_logger import ProcessRollingLogger
from langkit import llm_metrics # alternatively use 'light_metrics'
from whylogs.api.logger.experimental.logger.actor.time_util import Schedule, TimeGranularity
from transformers import GPT2Tokenizer, TextGenerationPipeline, GPT2LMHeadModel


# Initialize whylogs with your WhyLabs API key and target dataset ID. You can get an api key from the
# settings menu of you WhyLabs account. This loads credentials from the env directly
why.init()
# why.init(whylabs_api_key="KYE3amul9h.vKnLtOVp1IPcyAv7lEbgLEhGUrNirJ0RYEXVUhB6ZgSBHgaF7SbuF:org-JpsdM6", default_dataset_id="model-86")  # TODO delete
row_name = "image"

def create_logger():
    logger = ProcessRollingLogger(
        # This should match the model type in WhyLabs. We're using a daily model here.
        aggregate_by=TimeGranularity.Day,
        # The profiles will be uploaded from the rolling logger to WhyLabs every 5 minutes. Data
        # will accumulates during that time.
        write_schedule=Schedule(cadence=TimeGranularity.Minute, interval=5),
        schema=llm_metrics.init(),
    )

    logger.start()
    return logger


logger = create_logger()


def model_fn(model_dir):
    """
    Load the model for inference
    """

    # Load GPT2 tokenizer from disk.
    vocab_path = os.path.join(model_dir, 'model/vocab.json')
    merges_path = os.path.join(model_dir, 'model/merges.txt')
    
    tokenizer = GPT2Tokenizer(vocab_file=vocab_path,
                              merges_file=merges_path)

    # Load GPT2 model from disk.
    model_path = os.path.join(model_dir, 'model/')
    model = GPT2LMHeadModel.from_pretrained(model_path)

    return TextGenerationPipeline(model=model, tokenizer=tokenizer)


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    return model.__call__(input_data)

def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/json'
    return json.loads(request_body)

def predict_fn(input_data, model):
    if 'flush' in input_data:
        # Utility for flushing the logger, which forces it to upload any pending profiles synchronously.
        print('>> flushing')
        logger.flush()
        return 'flushed'

    if 'close' in input_data:
        print('>> closing')
        logger.close()
        return 'closed'

    output = model.__call__(input_data, max_length=100)

    try:
        # Log image async with whylogs. This won't hold up predictions.
        print(f'>> prompt: ${input_data}')
        logger.log({'prompt': input_data, 'response': output})  
    except Exception as e:
        print(f"Failed to log image: {e}")
        print(traceback.format_exc())

    return output

def output_fn(prediction, content_type):
    return str(prediction)
