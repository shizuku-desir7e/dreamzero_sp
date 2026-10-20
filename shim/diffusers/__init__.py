"""Minimal diffusers shim: only ConfigMixin / ModelMixin / register_to_config.
Avoids installing diffusers (which conflicts with the pinned huggingface_hub
in the LightningRL env). We only need to CONSTRUCT the model, not save/load.
"""
from .configuration_utils import ConfigMixin, register_to_config
from .modeling_utils import ModelMixin
__all__ = ["ConfigMixin", "register_to_config", "ModelMixin"]
__version__ = "0.0.0-shim"
