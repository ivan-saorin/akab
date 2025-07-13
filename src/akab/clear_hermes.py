"""ClearHermes - Non-blinded execution for Level 1 Quick Compare"""

import json
import os
import time
from typing import Dict, Any, Optional
from substrate import HermesExecutor, ExecutionRequest, ExecutionResult
import logging

logger = logging.getLogger(__name__)


class ClearHermes(HermesExecutor):
    """
    HermesExecutor implementation with NO blinding for Level 1 testing.
    
    Used for:
    - Quick comparisons where you want to see which provider gave which response
    - Debugging and exploration
    - Cases where blinding is not needed
    """
    
    def __init__(self):
        """Initialize with same providers as BlindedHermes but no blinding logic"""
        # Initialize provider clients
        self.providers = {}
        self._init_providers()
        
        # Validate we have at least one provider
        if not self.providers:
            raise RuntimeError(
                "No LLM providers available! Check your API keys in .env file."
            )
    
    def _init_providers(self):
        """Initialize LLM provider clients - same as BlindedHermes."""
        # Initialize Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                self.providers["anthropic"] = anthropic.AsyncAnthropic(api_key=anthropic_key)
                logger.info("Anthropic provider initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import anthropic: {e}")
                raise RuntimeError(
                    "Anthropic package not installed. Run: pip install anthropic"
                ) from e
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
                raise RuntimeError(f"Anthropic initialization failed: {e}") from e
        else:
            logger.warning("ANTHROPIC_API_KEY not found in environment")
        
        # Initialize OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                self.providers["openai"] = openai.AsyncOpenAI(api_key=openai_key)
                logger.info("OpenAI provider initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import openai: {e}")
                raise RuntimeError(
                    "OpenAI package not installed. Run: pip install openai"
                ) from e
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                raise RuntimeError(f"OpenAI initialization failed: {e}") from e
        else:
            logger.warning("OPENAI_API_KEY not found in environment")
            
        # Initialize Google
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                self.providers["google"] = genai
                logger.info("Google provider initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import google-generativeai: {e}")
                logger.warning("Google provider not available - install google-generativeai")
            except Exception as e:
                logger.error(f"Failed to initialize Google: {e}")
                logger.warning(f"Google provider not available: {e}")
        else:
            logger.warning("GOOGLE_API_KEY not found in environment")
    
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute with NO blinding - return clear provider/model information."""
        provider = request.parameters.get("provider", "").lower()
        
        if provider not in self.providers:
            return ExecutionResult(
                response="",
                model_id=request.model_id,
                model_name=request.model_name,  # Keep original model name
                metadata={
                    "error": f"Provider '{provider}' not available",
                    "available_providers": list(self.providers.keys())
                },
                execution_time=0,
                tokens_used=0,
                cost=0,
                error=f"Provider '{provider}' not available"
            )
        
        start_time = time.time()
        
        try:
            # Execute based on provider
            if provider == "anthropic":
                result = await self._execute_anthropic(request)
            elif provider == "openai":
                result = await self._execute_openai(request)
            elif provider == "google":
                result = await self._execute_google(request)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            execution_time = time.time() - start_time
            
            # Validate we got a real response
            if not result.response or result.response.strip() == "":
                return ExecutionResult(
                    response="",
                    model_id=request.model_id,
                    model_name=request.model_name,
                    metadata={
                        "error": "Empty response from provider",
                        "provider": provider
                    },
                    execution_time=execution_time,
                    tokens_used=0,
                    cost=0,
                    error="Empty response received"
                )
            
            # Update with clear metadata - NO BLINDING
            result.execution_time = execution_time
            result.model_name = request.model_name  # Keep original
            result.metadata = {
                **(result.metadata or {}),
                "provider": provider,
                "model": request.model_id,
                "clear_execution": True  # Flag for Level 1
            }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                response="",
                model_id=request.model_id,
                model_name=request.model_name,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "provider": provider
                },
                execution_time=execution_time,
                tokens_used=0,
                cost=0,
                error=str(e)
            )
    
    async def _execute_anthropic(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute with Anthropic API - same as BlindedHermes but no blinding."""
        client = self.providers["anthropic"]
        
        messages = [{"role": "user", "content": request.prompt}]
        max_tokens = request.constraints.get("max_tokens", 1000)
        temperature = request.constraints.get("temperature", 0.7)
        
        response = await client.messages.create(
            model=request.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Calculate cost
        pricing = {
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
        
        model_pricing = pricing.get(request.model_id, {"input": 0.003, "output": 0.015})
        input_cost = (response.usage.input_tokens / 1000) * model_pricing["input"]
        output_cost = (response.usage.output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return ExecutionResult(
            response=response.content[0].text,
            model_id=request.model_id,
            model_name=request.model_name,
            metadata={
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            execution_time=0,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            cost=total_cost
        )
    
    async def _execute_openai(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute with OpenAI API."""
        client = self.providers["openai"]
        
        messages = [{"role": "user", "content": request.prompt}]
        max_tokens = request.constraints.get("max_tokens", 1000)
        temperature = request.constraints.get("temperature", 0.7)
        
        response = await client.chat.completions.create(
            model=request.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Calculate cost
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
        }
        
        model_pricing = pricing.get(request.model_id, {"input": 0.01, "output": 0.03})
        input_cost = (response.usage.prompt_tokens / 1000) * model_pricing["input"]
        output_cost = (response.usage.completion_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return ExecutionResult(
            response=response.choices[0].message.content,
            model_id=request.model_id,
            model_name=request.model_name,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            },
            execution_time=0,
            tokens_used=response.usage.total_tokens,
            cost=total_cost
        )
    
    async def _execute_google(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute with Google API."""
        import asyncio
        genai = self.providers["google"]
        
        max_tokens = request.constraints.get("max_tokens", 1000)
        temperature = request.constraints.get("temperature", 0.7)
        
        model = genai.GenerativeModel(request.model_id)
        
        def _generate():
            response = model.generate_content(
                request.prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return response
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _generate)
        
        # Estimate tokens
        input_tokens = len(request.prompt.split()) * 1.3
        output_tokens = len(response.text.split()) * 1.3
        
        # Calculate cost
        pricing = {
            "gemini-1.5-flash": {"input": 0.00035, "output": 0.00105},
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.00375},
            "gemini-pro": {"input": 0.0005, "output": 0.0015}
        }
        
        model_pricing = pricing.get(request.model_id, {"input": 0.001, "output": 0.003})
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return ExecutionResult(
            response=response.text,
            model_id=request.model_id,
            model_name=request.model_name,
            metadata={
                "estimated_tokens": True,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens)
            },
            execution_time=0,
            tokens_used=int(input_tokens + output_tokens),
            cost=total_cost
        )
