"""
langextract — Google langextract + LLMFactory integration for Korean MMS entity extraction.

Bridges the existing LLMFactory (LangChain ChatOpenAI/ChatAnthropic) with Google's langextract
library for structured entity extraction from Korean MMS advertisement messages.

This package EXTENDS the pip-installed `langextract` via pkgutil.extend_path, so both the
pip package's submodules (core, providers, extraction, etc.) and our custom modules
(provider, extract, schemas, examples) are accessible under the `langextract` namespace.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from pkgutil import extend_path
from typing import Any, Dict

# --- Namespace merging ---
# Merge our directory with the pip-installed langextract's directory so that
# `langextract.core`, `langextract.providers` etc. resolve to the pip package,
# while `langextract.provider`, `langextract.extract` etc. resolve to our files.
__path__ = extend_path(__path__, __name__)

# --- sys.path setup ---
# Add mms_extractor_exp to sys.path so its modules (config.settings, utils.llm_factory)
# are importable by our provider.py
# Force mms_extractor_exp to front of sys.path so config.settings is found
# before any AgenticWorkflow/config.py file
_mms_exp_path = str(Path(__file__).parent.parent / "mms_extractor_exp")
if _mms_exp_path in sys.path:
    sys.path.remove(_mms_exp_path)
sys.path.insert(0, _mms_exp_path)

# --- Re-export pip langextract's public API ---
# These come from the pip-installed langextract (resolved via extended __path__)
from langextract.extraction import extract as _pip_extract
from langextract import visualization as _pip_viz

def extract(*args: Any, **kwargs: Any) -> Any:
    """Top-level API: lx.extract(...). Delegates to pip langextract."""
    return _pip_extract(*args, **kwargs)

def visualize(*args: Any, **kwargs: Any) -> Any:
    """Top-level API: lx.visualize(...)."""
    return _pip_viz.visualize(*args, **kwargs)

# --- Our MMS-specific exports ---
from langextract.extract_mms import extract_mms_entities  # noqa: E402
from langextract.provider import LangChainProvider  # noqa: E402

__all__ = [
    # Pip langextract API
    "extract",
    "visualize",
    # Our additions
    "extract_mms_entities",
    "LangChainProvider",
]

# --- PEP 562 lazy loading for pip submodules ---
_LAZY_MODULES = {
    "annotation": "langextract.annotation",
    "data": "langextract.data",
    "exceptions": "langextract.exceptions",
    "factory": "langextract.factory",
    "inference": "langextract.inference",
    "io": "langextract.io",
    "prompting": "langextract.prompting",
    "providers": "langextract.providers",
    "resolver": "langextract.resolver",
    "schema": "langextract.schema",
    "core": "langextract.core",
    "plugins": "langextract.plugins",
}

_CACHE: Dict[str, Any] = {}

def __getattr__(name: str) -> Any:
    if name in _CACHE:
        return _CACHE[name]
    modpath = _LAZY_MODULES.get(name)
    if modpath is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(modpath)
    sys.modules[f"{__name__}.{name}"] = module
    setattr(sys.modules[__name__], name, module)
    _CACHE[name] = module
    return module

def __dir__():
    return sorted(__all__ + list(_LAZY_MODULES.keys()))
