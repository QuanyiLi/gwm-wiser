import os
import glob
import yaml
from gwm_wiser import ASSET_ROOT


class ConfigRegistry:
    _configs = {}

    @classmethod
    def register_configs(cls):
        """Register all configs found in gwm_wiser/assets/configs/"""
        config_dir = os.path.join(ASSET_ROOT, "configs")
        pattern = os.path.join(config_dir, "config_*.yaml")

        for cfg_path in glob.glob(pattern):
            # Load the config
            with open(cfg_path, "r") as f:
                cfg_content = yaml.safe_load(f)

            # 1. Register by filename stem (e.g. "config_0")
            filename = os.path.basename(cfg_path)
            stem = os.path.splitext(filename)[0]
            cls._configs[stem] = cfg_content

            # 2. Register by internal name if present (e.g. "number")
            if "name" in cfg_content and cfg_content["name"] != "placeholder":
                cls._configs[cfg_content["name"]] = cfg_content

    @classmethod
    def get_config(cls, name):
        """Get a config by name (filename stem or internal name)."""
        # Lazy loading
        if not cls._configs:
            cls.register_configs()

        if name not in cls._configs:
            available = list(cls._configs.keys())
            # Limit message size if too many configs
            msg_available = available if len(available) < 10 else f"{available[:10]}..."
            raise ValueError(f"Config '{name}' not found. Available: {msg_available}")

        return cls._configs[name]

    @classmethod
    def get_all_names(cls):
        if not cls._configs:
            cls.register_configs()
        return list(cls._configs.keys())
