"""Load and validate YAML configuration files."""

from pathlib import Path
import yaml


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_yaml(path: Path) -> dict:
    """Read a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(config_dir: Path | None = None) -> dict:
    """Load config.yaml and models.yaml, returning a merged dict.

    Parameters
    ----------
    config_dir : Path, optional
        Directory containing config.yaml and models.yaml.
        Defaults to the ``config/`` directory at the repository root.

    Returns
    -------
    dict
        Merged configuration with keys from both files.
    """
    config_dir = Path(config_dir) if config_dir else _CONFIG_DIR
    cfg = load_yaml(config_dir / "config.yaml")
    models_cfg = load_yaml(config_dir / "models.yaml")
    cfg.update(models_cfg)
    return cfg


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent
