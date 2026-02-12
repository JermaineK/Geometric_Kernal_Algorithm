"""Domain adapters and registration helpers."""

from gka.domains.em_resonator import EMResonatorAdapter
from gka.domains.josephson import JosephsonAdapter
from gka.domains.oam import OAMAdapter
from gka.domains.plasma import PlasmaAdapter
from gka.domains.synthetic import SyntheticAdapter
from gka.domains.weather import WeatherAdapter

__all__ = [
    "register_builtin_adapters",
    "WeatherAdapter",
    "JosephsonAdapter",
    "OAMAdapter",
    "EMResonatorAdapter",
    "PlasmaAdapter",
    "SyntheticAdapter",
]


def register_builtin_adapters() -> None:
    from gka.core.registry import register_adapter

    register_adapter(WeatherAdapter())
    register_adapter(JosephsonAdapter())
    register_adapter(OAMAdapter())
    register_adapter(EMResonatorAdapter())
    register_adapter(PlasmaAdapter())
    register_adapter(SyntheticAdapter())
