"""

"""
import os
from enum import Enum
from pathlib import Path
from pydantic import SecretStr, Field
from google.cloud import secretmanager
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """Allowed log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"

class SecretManagerSettings:
    """Custom settings source for GCP Secret Manager"""

    def __init__(self):
        self.project_id = os.environ.get('PROJECT_ID')
        self.client = None

        if os.environ.get('ENVIRONMENT') == 'production':
            try:
                self.client = secretmanager.SecretManagerServiceClient()
            except Exception as e:
                print(f"Warning: Could not initialize Secret Manager client: {e}")

    def get_secret(self, secret_name: str) -> Optional[str]:
        if self.client and self.project_id:
            try:
                name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
                response = self.client.access_secret_version(request={"name": name})
                return response.payload.data.decode('UTF-8')
            except Exception as e:
                print(f"Error accessing secret {secret_name}: {e}")
                return None
        return None


secret_manager = SecretManagerSettings()


def secret_manager_settings(settings: BaseSettings) -> Dict[str, Any]:
    """Custom settings source that fetches from Secret Manager"""
    secrets = {}

    # Map field names to secret names in Secret Manager
    secret_mapping = {
        'openai_api_key': 'openai-api-key',
        'database_url': 'database-url',
    }

    for field_name, secret_name in secret_mapping.items():
        secret_value = secret_manager.get_secret(secret_name)
        if secret_value:
            secrets[field_name] = secret_value

    return secrets


class LangChainConfig(BaseSettings):
    """
    LangChain Backend Server Configuration

    This configuration supports multiple LLM providers, vector stores,
    and deployment environments with GCP Secret Manager integration.
    """

    # APPLICATION SETTINGS
    app_name: str = Field("simple-project", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    environment: str = Field("development", description="Environment (development/staging/production)")
    debug: bool = Field(False, description="Enable debug mode")

    # Server configuration
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, ge=1, le=65535, description="Server port")
    workers: int = Field(1, ge=1, le=32, description="Number of worker processes")

    # Logging
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    log_format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")

    # GCP SETTINGS
    project_id: str = Field("default-project-id", description="GCP Project ID")
    region: str = Field("us-central1", description="GCP Region")

    # LLM PROVIDER SETTINGS
    tavily_api_key: Optional[SecretStr] = Field(None, description="Tavily API key")
    primary_llm_provider: LLMProvider = Field(LLMProvider.OPENAI, description="Primary LLM provider")
    secondary_llm_provider:LLMProvider = Field(LLMProvider.GOOGLE, description="Primary LLM provider")

    # OpenAI
    openai_api_key: Optional[SecretStr] = Field(None, description="OpenAI API key")
    openai_model: str = Field("gpt-4", description="OpenAI model name")
    openai_temperature: float = Field(0.2, ge=0.0, le=2.0, description="OpenAI temperature")
    openai_top_p: float = Field(0.2, ge=0.0, le=1.0, description="OpenAI TOP P")
    openai_top_k: float = Field(20, ge=1, le=100, description="OpenAI TOP K")
    openai_max_tokens: float = Field(300, ge=0 , description="OpenAI max token")
    openai_frequency_penalty: float = Field(1, ge=1, le=2.0, description="OpenAI frequency penalty")


    # Gemini
    google_api_key: Optional[SecretStr] = Field(None, description="Google API key")
    google_model: str = Field("gemini-1.5-pro", description="Google model name")
    google_temperature: float = Field(0.2, ge=0.0, le=2.0, description="Google temperature")
    google_top_p: float = Field(0.2, ge=0.0, le=1.0, description="Google TOP P")
    google_top_k: int = Field(20, ge=1, le=100, description="Google TOP K")
    google_max_tokens: float = Field(300, ge=0 , description="Google max token")
    google_frequency_penalty: float = Field(1, ge=-2.0, le=2.0, description="Google frequency penalty")

    # DATABASE SETTINGS
    database_url: str = Field("sqlite:///./langchain_utils.db", description="Database connection URL")

    # Webhook security
    # webhook_secret: Optional[SecretStr] = Field(None, description="Webhook secret for validation")


    # HELPER METHODS
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration based on selected provider"""
        if self.llm_provider == LLMProvider.OPENAI:
            return {
                "provider": "openai",
                "api_key": self.openai_api_key.get_secret_value() if self.openai_api_key else None,
                "model": self.openai_model,
                "temperature": self.openai_temperature,
                "max_tokens": self.openai_max_tokens,
                "top_p": self.openai_top_p,
                "frequency_penalty": self.openai_frequency_penalty,
            }
        elif self.llm_provider == LLMProvider.GOOGLE:
            return {
                "provider": "google",
                "api_key": self.google_api_key.get_secret_value() if self.google_api_key else None,
                "model": self.google_model,
                "temperature": self.google_temperature,
                "top_p": self.google_top_p,
                "top_k": self.google_top_k,
                "max_tokens": self.google_max_tokens,
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": self.database_url,
            "echo": self.database_echo,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
        }

    class Config:
        env_file = Path(__file__).parent.parent.parent / '.env'
        env_file_encoding = 'utf-8'

        # Custom settings sources priority
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (
                init_settings,
                secret_manager_settings,
                env_settings,
                file_secret_settings,
            )


# Initialize configuration with validation
try:
    # noinspection PyArgumentList
    config = LangChainConfig()
    print(f"Project configuration loaded successfully.")

except Exception as e:
    print(f"Configuration error: {e}")
    raise


# Export config instance
__all__ = ['config', 'LangChainConfig']
