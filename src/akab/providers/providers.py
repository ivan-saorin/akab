"""
AKAB Provider Management
Handles multiple LLM providers with unified interface
"""
import os
import json
import asyncio
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

# Optional imports for remote providers
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

try:
    from instructor import Instructor, patch
    HAS_INSTRUCTOR = True
except ImportError:
    HAS_INSTRUCTOR = False

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Provider execution types"""
    LOCAL = "local"      # MCP client execution
    REMOTE = "remote"    # API-based execution


class Provider(ABC):
    """Base class for all providers"""
    
    def __init__(self, name: str, type: ProviderType):
        self.name = name
        self.type = type
        self.available = self.check_availability()
    
    @abstractmethod
    def check_availability(self) -> bool:
        """Check if provider is available"""
        pass
    
    @abstractmethod
    async def execute(
        self,
        prompt: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute prompt and return response with metadata"""
        pass
    
    @abstractmethod
    def estimate_cost(
        self,
        prompt_tokens: int,
        max_response_tokens: int = 1000
    ) -> float:
        """Estimate cost for execution"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": self.name,
            "type": self.type.value,
            "available": self.available
        }


class LocalProvider(Provider):
    """Local MCP execution provider"""
    
    def __init__(self):
        super().__init__("anthropic-local", ProviderType.LOCAL)
    
    def check_availability(self) -> bool:
        """Local provider is always available"""
        return True
    
    async def execute(
        self,
        prompt: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Local execution returns prompt for MCP client"""
        metadata = {
            "provider": self.name,
            "type": "local",
            "timestamp": datetime.now().isoformat(),
            "execution_mode": "mcp_client"
        }
        
        # For local, we return the prompt itself
        # The MCP client will execute it
        return prompt, metadata
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        max_response_tokens: int = 1000
    ) -> float:
        """Local execution is free"""
        return 0.0


class OpenAIProvider(Provider):
    """OpenAI API provider"""
    
    # Cost per 1K tokens (input/output)
    COSTS = {
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
    }
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        super().__init__(f"openai/{model}", ProviderType.REMOTE)
        
        if self.available:
            self.client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            if HAS_INSTRUCTOR:
                self.instructor_client = patch(self.client)
    
    def check_availability(self) -> bool:
        """Check if OpenAI is available"""
        return HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
    
    async def execute(
        self,
        prompt: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute via OpenAI API"""
        if not self.available:
            raise RuntimeError("OpenAI provider not available")
        
        config = config or {}
        
        try:
            start_time = datetime.now()
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2000)
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            content = response.choices[0].message.content
            
            metadata = {
                "provider": self.name,
                "model": self.model,
                "timestamp": end_time.isoformat(),
                "execution_time": execution_time,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": self._calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            }
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"OpenAI execution error: {e}")
            raise
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        max_response_tokens: int = 1000
    ) -> float:
        """Estimate cost for OpenAI execution"""
        return self._calculate_cost(prompt_tokens, max_response_tokens)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost"""
        model_key = self.model
        if model_key not in self.COSTS:
            # Default to GPT-3.5 costs
            model_key = "gpt-3.5-turbo"
        
        costs = self.COSTS[model_key]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return round(input_cost + output_cost, 4)


class AnthropicAPIProvider(Provider):
    """Anthropic API provider"""
    
    COSTS = {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
    }
    
    def __init__(self, model: str = "claude-3-sonnet"):
        self.model = model
        super().__init__(f"anthropic-api/{model}", ProviderType.REMOTE)
        
        if self.available:
            self.client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
    
    def check_availability(self) -> bool:
        """Check if Anthropic API is available"""
        return HAS_ANTHROPIC and bool(os.getenv("ANTHROPIC_API_KEY"))
    
    async def execute(
        self,
        prompt: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute via Anthropic API"""
        if not self.available:
            raise RuntimeError("Anthropic API provider not available")
        
        config = config or {}
        
        try:
            start_time = datetime.now()
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=config.get("max_tokens", 2000),
                temperature=config.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            content = response.content[0].text
            
            # Estimate tokens (Anthropic doesn't always provide usage)
            prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
            completion_tokens = len(content.split()) * 1.3
            
            metadata = {
                "provider": self.name,
                "model": self.model,
                "timestamp": end_time.isoformat(),
                "execution_time": execution_time,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
                "cost": self._calculate_cost(prompt_tokens, completion_tokens)
            }
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Anthropic API execution error: {e}")
            raise
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        max_response_tokens: int = 1000
    ) -> float:
        """Estimate cost for Anthropic execution"""
        return self._calculate_cost(prompt_tokens, max_response_tokens)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost"""
        model_key = self.model
        if model_key not in self.COSTS:
            model_key = "claude-3-sonnet"
        
        costs = self.COSTS[model_key]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return round(input_cost + output_cost, 4)


class GoogleProvider(Provider):
    """Google Gemini API provider"""
    
    COSTS = {
        "gemini-pro": {"input": 0.0005, "output": 0.0015}
    }
    
    def __init__(self, model: str = "gemini-pro"):
        self.model = model
        super().__init__(f"google/{model}", ProviderType.REMOTE)
        
        if self.available:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model_client = genai.GenerativeModel(model)
    
    def check_availability(self) -> bool:
        """Check if Google API is available"""
        return HAS_GOOGLE and bool(os.getenv("GOOGLE_API_KEY"))
    
    async def execute(
        self,
        prompt: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute via Google API"""
        if not self.available:
            raise RuntimeError("Google provider not available")
        
        config = config or {}
        
        try:
            start_time = datetime.now()
            
            # Google's async support is different
            response = await asyncio.to_thread(
                self.model_client.generate_content,
                prompt,
                generation_config={
                    "temperature": config.get("temperature", 0.7),
                    "max_output_tokens": config.get("max_tokens", 2000)
                }
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            content = response.text
            
            # Estimate tokens
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(content.split()) * 1.3
            
            metadata = {
                "provider": self.name,
                "model": self.model,
                "timestamp": end_time.isoformat(),
                "execution_time": execution_time,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
                "cost": self._calculate_cost(prompt_tokens, completion_tokens)
            }
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Google execution error: {e}")
            raise
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        max_response_tokens: int = 1000
    ) -> float:
        """Estimate cost for Google execution"""
        return self._calculate_cost(prompt_tokens, max_response_tokens)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost"""
        costs = self.COSTS.get(self.model, self.COSTS["gemini-pro"])
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return round(input_cost + output_cost, 4)


class ProviderManager:
    """Manages all available providers"""
    
    def __init__(self):
        self.providers: Dict[str, Provider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        # Always add local provider
        local_provider = LocalProvider()
        self.providers[local_provider.name] = local_provider
        
        # Add OpenAI providers
        for model in ["gpt-4-turbo", "gpt-3.5-turbo"]:
            provider = OpenAIProvider(model)
            if provider.available:
                self.providers[provider.name] = provider
        
        # Add Anthropic API providers
        for model in ["claude-3-opus", "claude-3-sonnet"]:
            provider = AnthropicAPIProvider(model)
            if provider.available:
                self.providers[provider.name] = provider
        
        # Add Google provider
        provider = GoogleProvider()
        if provider.available:
            self.providers[provider.name] = provider
        
        logger.info(f"Initialized {len(self.providers)} providers")
    
    def get_provider(self, name: str) -> Optional[Provider]:
        """Get provider by name"""
        return self.providers.get(name)
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all available providers"""
        return [p.get_info() for p in self.providers.values()]
    
    def get_remote_providers(self) -> List[str]:
        """Get list of remote provider names"""
        return [
            name for name, provider in self.providers.items()
            if provider.type == ProviderType.REMOTE
        ]
    
    async def execute(
        self,
        provider_name: str,
        prompt: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute prompt with specified provider"""
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        if not provider.available:
            raise RuntimeError(f"Provider '{provider_name}' not available")
        
        return await provider.execute(prompt, config)
    
    async def estimate_campaign_cost(
        self,
        experiments: List[Dict[str, Any]]
    ) -> float:
        """Estimate total cost for campaign experiments"""
        total_cost = 0.0
        
        for exp in experiments:
            provider_name = exp.get("provider", "anthropic-local")
            provider = self.get_provider(provider_name)
            
            if provider and provider.type == ProviderType.REMOTE:
                # Estimate based on average prompt size
                prompt_tokens = exp.get("estimated_tokens", 500)
                cost = provider.estimate_cost(prompt_tokens)
                total_cost += cost
        
        return round(total_cost, 2)
