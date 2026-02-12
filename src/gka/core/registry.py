"""Adapter registry for domain-agnostic pipeline execution."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gka.domains.base import DomainAdapter


_ADAPTERS: dict[str, "DomainAdapter"] = {}


class RegistryError(KeyError):
    """Raised when an adapter cannot be resolved."""


def register_adapter(adapter: "DomainAdapter") -> None:
    name = adapter.name.strip().lower()
    if not name:
        raise RegistryError("Adapter name cannot be empty")
    _ADAPTERS[name] = adapter


def get_adapter(name: str) -> "DomainAdapter":
    key = name.strip().lower()
    if key in _ADAPTERS:
        return _ADAPTERS[key]
    _load_entrypoint_adapters()
    if key not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS)) or "none"
        raise RegistryError(f"Unknown domain '{name}'. Available domains: {available}")
    return _ADAPTERS[key]


def available_domains() -> list[str]:
    _load_entrypoint_adapters()
    return sorted(_ADAPTERS)


def _load_entrypoint_adapters() -> None:
    eps = entry_points()
    group = eps.select(group="gka.domains") if hasattr(eps, "select") else eps.get("gka.domains", [])
    for ep in group:
        _load_entrypoint(ep)


def _load_entrypoint(ep: EntryPoint | Any) -> None:
    try:
        obj = ep.load()
    except Exception:
        return
    if callable(obj):
        inst = obj()
    else:
        inst = obj
    if hasattr(inst, "name"):
        register_adapter(inst)


def clear_registry() -> None:
    _ADAPTERS.clear()


def import_and_register(module_path: str, factory_name: str = "register") -> None:
    """Import a module and invoke its registration function."""

    module = import_module(module_path)
    factory = getattr(module, factory_name)
    factory()
