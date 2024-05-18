"""Settings for GenAI Graph Processor"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Base settings object"""

    log_level: str = "INFO"
    model: str = "llava-hf/llava-1.5-7b-hf"

    image_name: str = "data/images/line_plot.png"
    prompt_id: str = "1"

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False
    )
