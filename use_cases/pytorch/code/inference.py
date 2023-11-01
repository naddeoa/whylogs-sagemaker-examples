import traceback
import torch
import requests
import json

import whylogs as why
from whylogs.api.logger.experimental.logger.actor.process_rolling_logger import ProcessRollingLogger
from whylogs.api.logger.experimental.logger.actor.time_util import Schedule, TimeGranularity
from whylogs.extras.image_metric import init_image_schema


# Initialize whylogs with your WhyLabs API key and target dataset ID. You can get an api key from the
# settings menu of you WhyLabs account.
why.init() # This loads credentials from the env directly
row_name = "image"


def create_logger():
    logger = ProcessRollingLogger(
        # This should match the model type in WhyLabs. We're using a daily model here.
        aggregate_by=TimeGranularity.Day,
        # The profiles will be uploaded from the rolling logger to WhyLabs every 5 minutes. Data
        # will accumulates during that time.
        write_schedule=Schedule(cadence=TimeGranularity.Minute, interval=5),
        schema=init_image_schema(row_name),  # Enables image metrics
    )

    logger.start()
    return logger


# Utility function for converting our resnet class predictions into english.
def create_class_names():
    url = "https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt"
    response = requests.get(url)
    class_names = response.text.split("\n")
    return {i: class_names[i] for i in range(len(class_names))}


class_names = create_class_names()
logger = create_logger()


def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/resnet.pt")
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/json'
    body = json.loads(request_body)

    if 'flush' in body and body['flush']:
        # Utility for flushing the logger, which forces it to upload any pending profiles synchronously.
        logger.flush()
        return None


    if 'close' in body and body['close']:
        logger.close()
        return None

    # We're going to be uploading the preprocessed and original images to sagemaker for this example
    # to avoid having to deploy torchvision. We don't want to log the preprocessed image, just the actual one.
    assert 'image' in body
    assert 'raw_img' in body

    try:
        # Log image async with whylogs. This won't hold up predictions.
        data = body['raw_img']
        print(f'logging type {type(data)} {type(data[0][0][0])}')
        logger.log({row_name: data})  
    except Exception as e:
        print(f"Failed to log image: {e}")
        print(traceback.format_exc())

    return torch.tensor(body['image'], dtype=torch.float32)


def predict_fn(input_tensor: torch.Tensor, model: torch.nn.Module):
    if input_tensor is None:
        return ""

    img_batch = torch.unsqueeze(input_tensor, 0)
    with torch.no_grad():
        output_tensor = model(img_batch)

    _, predicted_class = torch.max(output_tensor, 1)
    predicted_label = class_names[float(predicted_class.numpy())]
    return predicted_label


def output_fn(prediction, content_type):
    return str(prediction)
