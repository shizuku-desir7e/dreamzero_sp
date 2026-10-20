import torch.nn as nn
from ..configuration_utils import ConfigMixin

class ModelMixin(ConfigMixin, nn.Module):
    def save_pretrained(self, *a, **k):
        raise NotImplementedError("shim")
    @classmethod
    def from_pretrained(cls, *a, **k):
        raise NotImplementedError("shim")
