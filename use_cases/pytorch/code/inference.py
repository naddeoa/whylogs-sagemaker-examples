import traceback
import torch
import requests
import json

from typing import Dict
from PIL import Image
import whylogs as why
from whylogs.api.logger.experimental.logger.actor.process_rolling_logger import ProcessRollingLogger
from whylogs.api.logger.experimental.logger.actor.time_util import Schedule, TimeGranularity
from whylogs.core.resolvers import Resolver
from whylogs.core.schema import ColumnSchema, DatasetSchema
from whylogs.core.datatypes import DataType
from whylogs.extras.image_metric import ImageMetric, ImageMetricConfig
from whylogs.core.metrics.metrics import Metric

why.init(whylabs_api_key="3QYTCJSgE0.ME8gmCfqmdAd233vet4WuLs4eScPW0Lxwb4jHUglnIyQOXZjx5vLA:org-JpsdM6", default_dataset_id="model-70")
# why.init()

row_name = "image"

class ImageResolver(Resolver):
    def resolve(self, name: str, why_type: DataType, column_schema: ColumnSchema) -> Dict[str, Metric]:
        return {ImageMetric.get_namespace(): ImageMetric.zero(column_schema.cfg)}

schema = DatasetSchema(
    types={row_name: Image.Image}, default_configs=ImageMetricConfig(), resolvers=ImageResolver()
)

logger = ProcessRollingLogger(
    aggregate_by=TimeGranularity.Day,
    write_schedule=Schedule(cadence=TimeGranularity.Minute, interval=5),
    schema=schema
)

logger.start()


def create_class_names():
    url = "https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt"
    response = requests.get(url)
    class_names = response.text.split("\n")
    return {i: class_names[i] for i in range(len(class_names))}


class_names = create_class_names()


def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/resnet.pt")
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/json'
    body = json.loads(request_body)
    assert 'image' in body

    image = body['image']

    try:
        print(f'Logging images {type(image)}')
        logger.log({row_name: image})  # Log async with whylogs
    except Exception as e:
        print(f"Failed to log image: {e}")
        print(traceback.format_exc())

    image = torch.tensor(image, dtype=torch.float32)
    return image


def predict_fn(input_tensor: torch.Tensor, model: torch.nn.Module):
    """
    Args:
        input_object (torch.Tensor): input image, not normalized
        model (torch.nn.Module): model to use for inference
    """
    img_batch = torch.unsqueeze(input_tensor, 0)
    with torch.no_grad():
        output_tensor = model(img_batch)

    _, predicted_class = torch.max(output_tensor, 1)
    predicted_label = class_names[float(predicted_class.numpy())]

    return predicted_label


def output_fn(prediction, content_type):
    return str(prediction)
