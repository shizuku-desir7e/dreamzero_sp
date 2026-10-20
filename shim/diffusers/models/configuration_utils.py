import functools, inspect

def register_to_config(init):
    sig = inspect.signature(init)
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        cfg = {k: v for k, v in bound.arguments.items() if k != "self"}
        object.__setattr__(self, "_internal_config", cfg)
        return init(self, *args, **kwargs)
    return wrapper

class ConfigMixin:
    def __init__(self, *args, **kwargs):
        super().__init__()          # MRO 继续走到 nn.Module

    @property
    def config(self):
        return getattr(self, "_internal_config", {})

    def register_to_config(self, **kw):
        self._internal_config.update(kw)

    @classmethod
    def from_config(cls, config, **overrides):
        cfg = dict(config); cfg.update(overrides)
        return cls(**cfg)

    def save_pretrained(self, *a, **k):
        raise NotImplementedError("shim")
