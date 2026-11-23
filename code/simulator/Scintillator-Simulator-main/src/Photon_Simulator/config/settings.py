from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    app_env: str = os.getenv("APP_ENV", "production")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    logger_name: str = os.getenv("LOGGER_NAME", "phSim")
    
settings = Settings()

