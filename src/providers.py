"""Provider management for AKAB - handles LLM provider abstraction."""

import os
import logging
from typing import Any, Dict, List, Optional, Protocol

# Import substrate components
try:
    from substrate import ProviderConfig, ValidationError
except ImportError:
    try:
        from .substrate import ProviderConfig, ValidationError
    except ImportError:
        from .substrate_stub import ProviderConfig, ValidationError

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete a prompt and return response."""
        ...


class ProviderManager:
    """Manages LLM providers with size-based naming."""
    
    # Default provider configurations
    DEFAULT_PROVIDERS = {
        # Anthropic
        "anthropic_xs": ProviderConfig(
            name="anthropic_xs",
            size="xs",
            model="claude-3-haiku-20240307",
            api_key_env="ANTHROPIC_API_KEY",
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
            max_tokens=4096,
        ),
        "anthropic_s": ProviderConfig(
            name="anthropic_s",
            size="s",
            model="claude-3-haiku-20240307",
            api_key_env="ANTHROPIC_API_KEY",
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
            max_tokens=4096,
        ),
        "anthropic_m": ProviderConfig(
            name="anthropic_m",
            size="m",
            model="claude-3-5-sonnet-20241022",
            api_key_env="ANTHROPIC_API_KEY",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            max_tokens=8192,
        ),
        "anthropic_l": ProviderConfig(
            name="anthropic_l",
            size="l",
            model="claude-3-5-sonnet-20241022",
            api_key_env="ANTHROPIC_API_KEY",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            max_tokens=8192,
        ),
        "anthropic_xl": ProviderConfig(
            name="anthropic_xl",
            size="xl",
            model="claude-3-opus-20240229",
            api_key_env="ANTHROPIC_API_KEY",
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            max_tokens=4096,
        ),
        
        # OpenAI
        "openai_s": ProviderConfig(
            name="openai_s",
            size="s",
            model="gpt-3.5-turbo",
            api_key_env="OPENAI_API_KEY",
            cost_per_1k_input=0.0005,
            cost_per_1k_output=0.0015,
            max_tokens=4096,
        ),
        "openai_m": ProviderConfig(
            name="openai_m",
            size="m",
            model="gpt-4",
            api_key_env="OPENAI_API_KEY",
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06,
            max_tokens=8192,
        ),
        "openai_l": ProviderConfig(
            name="openai_l",
            size="l",
            model="gpt-4-turbo",
            api_key_env="OPENAI_API_KEY",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            max_tokens=128000,
        ),
        
        # Google
        "google_s": ProviderConfig(
            name="google_s",
            size="s",
            model="gemini-1.5-flash",
            api_key_env="GOOGLE_API_KEY",
            cost_per_1k_input=0.00035,
            cost_per_1k_output=0.00105,
            max_tokens=8192,
        ),
        "google_m": ProviderConfig(
            name="google_m",
            size="m",
            model="gemini-1.5-pro",
            api_key_env="GOOGLE_API_KEY",
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.00375,
            max_tokens=8192,
        ),
    }
    
    def __init__(self, custom_providers: Optional[Dict[str, ProviderConfig]] = None):
        """Initialize provider manager.
        
        Args:
            custom_providers: Optional custom provider configurations
        """
        self.providers = self.DEFAULT_PROVIDERS.copy()
        if custom_providers:
            self.providers.update(custom_providers)
            
        self._provider_instances: Dict[str, LLMProvider] = {}
        
    def list_providers(self) -> List[str]:
        """List all available provider names."""
        return sorted(self.providers.keys())
        
    def is_valid_provider(self, provider_name: str) -> bool:
        """Check if provider name is valid."""
        return provider_name in self.providers
        
    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """Get configuration for a provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider configuration
            
        Raises:
            ValidationError: If provider not found
        """
        if not self.is_valid_provider(provider_name):
            raise ValidationError(
                f"Unknown provider: {provider_name}",
                field="provider",
                suggestions=self.list_providers()
            )
            
        return self.providers[provider_name]
        
    async def get_provider(self, provider_name: str) -> LLMProvider:
        """Get or create provider instance.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider instance
            
        Raises:
            ValidationError: If provider not found or not configured
        """
        if provider_name in self._provider_instances:
            return self._provider_instances[provider_name]
            
        config = self.get_provider_config(provider_name)
        
        # Check for API key
        api_key = os.getenv(config.api_key_env) if config.api_key_env else None
        if not api_key and config.api_key_env:
            raise ValidationError(
                f"API key not found for {provider_name}",
                field="api_key",
                suggestions=[
                    f"Set {config.api_key_env} environment variable",
                    "Use a .env file with python-dotenv",
                ]
            )
            
        # Create provider instance based on type
        if provider_name.startswith("anthropic"):
            provider = await self._create_anthropic_provider(config, api_key)
        elif provider_name.startswith("openai"):
            provider = await self._create_openai_provider(config, api_key)
        elif provider_name.startswith("google"):
            provider = await self._create_google_provider(config, api_key)
        else:
            raise ValidationError(
                f"Provider type not implemented: {provider_name}",
                field="provider"
            )
            
        self._provider_instances[provider_name] = provider
        return provider
        
    async def _create_anthropic_provider(
        self,
        config: ProviderConfig,
        api_key: Optional[str]
    ) -> LLMProvider:
        """Create Anthropic provider instance."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ValidationError(
                "Anthropic package not installed",
                suggestions=["pip install anthropic"]
            )
            
        class AnthropicProvider:
            def __init__(self, client, model):
                self.client = client
                self.model = model
                
            async def complete(self, prompt, **kwargs):
                response = await self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", 1000),
                    temperature=kwargs.get("temperature", 0.7),
                )
                return {
                    "text": response.content[0].text,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                }
                
        client = AsyncAnthropic(api_key=api_key)
        return AnthropicProvider(client, config.model)
        
    async def _create_openai_provider(
        self,
        config: ProviderConfig,
        api_key: Optional[str]
    ) -> LLMProvider:
        """Create OpenAI provider instance."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ValidationError(
                "OpenAI package not installed",
                suggestions=["pip install openai"]
            )
            
        class OpenAIProvider:
            def __init__(self, client, model):
                self.client = client
                self.model = model
                
            async def complete(self, prompt, **kwargs):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", 1000),
                    temperature=kwargs.get("temperature", 0.7),
                )
                return {
                    "text": response.choices[0].message.content,
                    "usage": {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    }
                }
                
        client = AsyncOpenAI(api_key=api_key)
        return OpenAIProvider(client, config.model)
        
    async def _create_google_provider(
        self,
        config: ProviderConfig,
        api_key: Optional[str]
    ) -> LLMProvider:
        """Create Google provider instance."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ValidationError(
                "Google Generative AI package not installed",
                suggestions=["pip install google-generativeai"]
            )
            
        class GoogleProvider:
            def __init__(self, model):
                self.model = model
                
            async def complete(self, prompt, **kwargs):
                # Google's API is sync, so we run in executor
                import asyncio
                
                def _generate():
                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            "max_output_tokens": kwargs.get("max_tokens", 1000),
                            "temperature": kwargs.get("temperature", 0.7),
                        }
                    )
                    return response
                    
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, _generate)
                
                # Estimate tokens (Google doesn't provide exact counts)
                input_tokens = len(prompt.split()) * 1.3
                output_tokens = len(response.text.split()) * 1.3
                
                return {
                    "text": response.text,
                    "usage": {
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                    }
                }
                
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.model)
        return GoogleProvider(model)
        
    def estimate_cost(
        self,
        provider_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Estimate cost for a completion.
        
        Args:
            provider_name: Name of the provider
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in dollars
        """
        config = self.get_provider_config(provider_name)
        
        input_cost = (input_tokens / 1000) * config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * config.cost_per_1k_output
        
        return input_cost + output_cost
        
    def get_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """Get detailed information about a provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider information dict
        """
        config = self.get_provider_config(provider_name)
        
        return {
            "name": config.name,
            "size": config.size,
            "model": config.model,
            "max_tokens": config.max_tokens,
            "cost_per_1k_input": config.cost_per_1k_input,
            "cost_per_1k_output": config.cost_per_1k_output,
            "supports_streaming": config.supports_streaming,
            "supports_tools": config.supports_tools,
            "api_key_configured": bool(
                os.getenv(config.api_key_env) if config.api_key_env else False
            ),
        }
