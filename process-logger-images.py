from PIL import Image
import numpy as np
from typing import Dict

import whylogs as why
from whylogs.api.logger.experimental.logger.actor.process_rolling_logger import ProcessRollingLogger
from whylogs.api.logger.experimental.logger.actor.time_util import Schedule, TimeGranularity
from whylogs.core.resolvers import Resolver
from whylogs.core.schema import ColumnSchema, DatasetSchema
from whylogs.core.datatypes import DataType
from whylogs.extras.image_metric import ImageMetric, ImageMetricConfig
from whylogs.core.metrics.metrics import Metric

why.init(api_key="3QYTCJSgE0.ME8gmCfqmdAd233vet4WuLs4eScPW0Lxwb4jHUglnIyQOXZjx5vLA:org-JpsdM6", default_dataset_id="model-69")

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

image = np.array(Image.open("/home/anthony/Downloads/dutch_oven.jpeg"))
print(image.shape)
try:
    logger.log({row_name: image.tolist()})
except Exception as e:
    print(e)

image = np.array(Image.open("/home/anthony/Downloads/Tiger_shark.png").convert("RGBA"))
print(image.shape)
try:
    logger.log({row_name: image.tolist()})
except Exception as e:
    print(e)

logger.close()
