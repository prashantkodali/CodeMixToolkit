"""
Configuration management for CodeMix Toolkit
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


class Config:
    """
    Configuration singleton for CodeMix Toolkit.

    Loads .env and config.yaml file if they exist in the current working directory.

    If no files are found, will use empty strings for the configuration values.

    Can also be updated dynamically using the update method.

    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # API Keys - empty by default
        self.OPENAI_API_KEY = ""
        self.HUGGINGFACE_API_KEY = ""
        self.GOOGLE_API_KEY = ""
        self.OPENROUTER_API_KEY = ""
        self.HUGGINGFACE_API_KEY = ""

        # Library configuration with defaults
        self.LOG_LEVEL = "INFO"

        self._initialized = True

        # if .env file exists, load it
        if os.path.exists(".env"):
            self.load_env(Path(".env"))

        # if config.yaml file exists, load it
        if os.path.exists("config.yaml"):
            self.load_yaml(Path("config.yaml"))

    def load_env(self, env_file: Path) -> None:
        """Load configuration from a .env file.

        Args:
            env_file: Path to .env file
        """
        if isinstance(env_file, str):
            env_file = Path(env_file)

        if not env_file.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        load_dotenv(env_file)

        # Update configuration from environment variables
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", self.OPENAI_API_KEY)
        self.HUGGINGFACE_API_KEY = os.getenv(
            "HUGGINGFACE_API_KEY", self.HUGGINGFACE_API_KEY
        )
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", self.GOOGLE_API_KEY)
        self.OPENROUTER_API_KEY = os.getenv(
            "OPENROUTER_API_KEY", self.OPENROUTER_API_KEY
        )
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)

    def load_yaml(self, yaml_file: Path) -> None:
        """Load configuration from a YAML file.

        Args:
            yaml_file: Path to YAML file
        """
        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_file}")

        with open(yaml_file) as f:
            config_data = yaml.safe_load(f)

        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def update(self, **kwargs) -> None:
        """Update individual configuration values.

        Args:
            **kwargs: Configuration key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")

    def validate(self, require_api_keys: bool = False) -> None:
        """Validate configuration settings.

        Args:
            require_api_keys: If True, will check that API keys are not empty
        """
        if require_api_keys:
            required_vars = ["OPENAI_API_KEY", "HUGGINGFACE_API_KEY", "GOOGLE_API_KEY"]
            missing = [var for var in required_vars if not getattr(self, var)]
            if missing:
                raise ValueError(f"Missing required API keys: {', '.join(missing)}")


# Create global config instance -- singleton of Config class
config = Config()
