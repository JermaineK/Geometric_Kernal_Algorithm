"""External adapter layer for messy real-world sources."""

from gka.adapters.base import AdapterOutput, DataAdapter
from gka.adapters.generic_image import GenericImageAdapter
from gka.adapters.generic_timeseries import GenericTimeseriesAdapter
from gka.adapters.weather_era5 import WeatherERA5Adapter

__all__ = [
    "AdapterOutput",
    "DataAdapter",
    "GenericImageAdapter",
    "GenericTimeseriesAdapter",
    "WeatherERA5Adapter",
]
