"""BlindedHermes - Triple-blinded execution for A/B testing"""

import hashlib
import json
import os
import re
import time
from typing import Dict, Any, Optional, Tuple, List
import aiofiles
from substrate import HermesExecutor, ExecutionRequest, ExecutionResult
import logging

logger = logging.getLogger(__name__)


class BlindedHermes(HermesExecutor):
    """
    HermesExecutor implementation with triple blinding for A/B testing.
    
    Triple Blinding:
    1. User Blinding: Hide model identities from user (variant IDs)
    2. Execution Blinding: No model names in logs or error messages
    3. Response Sanitization: Strip model indicators from LLM responses
    """
    
    def __init__(self, data_dir: str, krill_dir: str = None):
        self.data_dir = data_dir
        # Store blinding in krill for complete isolation
        if krill_dir:
            self.blinding_dir = os.path.join(krill_dir, "blinding")
        else:
            self.blinding_dir = os.path.join(data_dir, "blinding")
        
        # Ensure blinding directory exists
        os.makedirs(self.blinding_dir, exist_ok=True)
        
        # Initialize provider clients (FAIL LOUDLY if not available)
        self.providers = {}
        self._init_providers()
        
        # Validate we have at least one provider
        if not self.providers:
            raise RuntimeError(
                "No LLM providers available! Check your API keys in .env file."
            )
    
    def _init_providers(self):
        """Initialize LLM provider clients - FAIL LOUDLY on errors."""
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
                # Google is optional, just warn
                logger.warning("Google provider not available - install google-generativeai")
            except Exception as e:
                logger.error(f"Failed to initialize Google: {e}")
                logger.warning(f"Google provider not available: {e}")
        else:
            logger.warning("GOOGLE_API_KEY not found in environment")
    
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute with triple blinding."""
        # Generate blinding ID
        blinding_id = self._generate_blinding_id(request)
        
        # Store blinding mapping
        await self._store_blinding(blinding_id, request)
        
        # Execute with blinded reference
        result = await self._execute_blinded(request, blinding_id)
        
        # Apply Level 3 blinding: Sanitize response
        if result.response:
            result.response = self._sanitize_response(result.response)
        
        # Return result with blinded ID
        return ExecutionResult(
            response=result.response,
            model_id=blinding_id,  # Level 1: Return blinded ID
            model_name=None,  # Level 1: Hide model name
            metadata={
                **(result.metadata or {}),
                "blinded_id": blinding_id,
                "blinded": True
            },
            execution_time=result.execution_time,
            tokens_used=result.tokens_used,
            cost=result.cost,
            error=result.error
        )
    
    def _generate_blinding_id(self, request: ExecutionRequest) -> str:
        """Generate deterministic blinding ID."""
        # Use campaign ID + variant ID + iteration for uniqueness
        components = [
            request.parameters.get("campaign_id", ""),
            request.parameters.get("variant_id", ""),
            str(request.parameters.get("iteration", "")),
            str(time.time())
        ]
        
        # Create hash
        hash_input = "|".join(components)
        hash_obj = hashlib.sha256(hash_input.encode())
        
        # Take first 6 chars for readability
        return f"variant_{hash_obj.hexdigest()[:6]}"
    
    async def _store_blinding(self, blinding_id: str, request: ExecutionRequest):
        """Store blinding mapping for later unblinding."""
        mapping = {
            "blinding_id": blinding_id,
            "model_id": request.model_id,
            "model_name": request.model_name,
            "provider": request.parameters.get("provider"),
            "campaign_id": request.parameters.get("campaign_id"),
            "variant_id": request.parameters.get("variant_id"),
            "iteration": request.parameters.get("iteration"),
            "timestamp": time.time()
        }
        
        # Store to file
        filepath = os.path.join(self.blinding_dir, f"{blinding_id}.json")
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(mapping, indent=2))
    
    def _sanitize_response(self, response: str) -> str:
        """Level 3 blinding: Remove model self-identification from responses."""
        if not response:
            return response
            
        # Patterns to remove model self-identification
        patterns = [
            # Claude patterns
            r"(?i)as claude.*?,?\s*",
            r"(?i)i'?m claude.*?,?\s*",
            r"(?i)my name is claude.*?,?\s*",
            r"(?i)i am claude.*?,?\s*",
            r"(?i)as an anthropic.*?,?\s*",
            # GPT patterns
            r"(?i)as (?:chatgpt|gpt-?\d+(?:\.\d+)?|an openai.*?),?\s*",
            r"(?i)i'?m (?:chatgpt|gpt-?\d+(?:\.\d+)?|an openai.*?),?\s*",
            r"(?i)my name is (?:chatgpt|gpt-?\d+(?:\.\d+)?),?\s*",
            r"(?i)i am (?:chatgpt|gpt-?\d+(?:\.\d+)?|an openai.*?),?\s*",
            # Gemini patterns
            r"(?i)as (?:gemini|bard|a google.*?),?\s*",
            r"(?i)i'?m (?:gemini|bard|a google.*?),?\s*",
            r"(?i)my name is (?:gemini|bard),?\s*",
            r"(?i)i am (?:gemini|bard|a google.*?),?\s*",
            # Generic AI patterns
            r"(?i)as an? (?:ai|artificial intelligence) (?:assistant|model|system) (?:created|developed|trained) by (?:anthropic|openai|google),?\s*",
        ]
        
        sanitized = response
        for pattern in patterns:
            sanitized = re.sub(pattern, "", sanitized)
            
        # Clean up any double spaces or leading punctuation
        sanitized = re.sub(r'\s+', ' ', sanitized)
        sanitized = re.sub(r'^\s*[,;]\s*', '', sanitized)
        
        return sanitized.strip()
    
    async def _execute_blinded(self, request: ExecutionRequest, blinding_id: str) -> ExecutionResult:
        """Execute the actual request with real LLM providers."""
        provider = request.parameters.get("provider", "").lower()
        
        # Level 2 blinding: Don't expose provider in error messages
        if provider not in self.providers:
            available_count = len(self.providers)
            return ExecutionResult(
                response="",
                model_id=request.model_id,
                model_name=None,  # Level 2: Hide model name
                metadata={
                    "error": f"Provider not available ({available_count} providers configured)",
                    "blinded_error": True
                },
                execution_time=0,
                tokens_used=0,
                cost=0,
                error=f"Provider not available"
            )
        
        start_time = time.time()
        
        try:
            if provider == "anthropic":
                result = await self._execute_anthropic(request)
            elif provider == "openai":
                result = await self._execute_openai(request)
            elif provider == "google":
                result = await self._execute_google(request)
            else:
                raise ValueError(f"Unsupported provider")
            
            execution_time = time.time() - start_time
            
            # Validate we got a real response
            if not result.response or result.response.strip() == "":
                return ExecutionResult(
                    response="",
                    model_id=request.model_id,
                    model_name=None,
                    metadata={
                        "error": "Empty response from provider",
                        "blinded_error": True
                    },
                    execution_time=execution_time,
                    tokens_used=0,
                    cost=0,
                    error="Empty response received"
                )
            
            # Update result with execution time
            result.execution_time = execution_time
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            # Level 2 blinding: Generic error messages
            error_msg = self._sanitize_error(str(e))
            
            return ExecutionResult(
                response="",
                model_id=request.model_id,
                model_name=None,  # Level 2: Hide model name
                metadata={
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "blinded_error": True
                },
                execution_time=execution_time,
                tokens_used=0,
                cost=0,
                error=error_msg
            )
    
    def _sanitize_error(self, error: str) -> str:
        """Level 2 blinding: Remove provider/model details from error messages."""
        # Remove specific model names
        sanitized = re.sub(r'(?i)(claude|gpt-?\d+|gemini|anthropic|openai|google)', 'model', error)
        # Remove API keys if accidentally included
        sanitized = re.sub(r'(api_key|key)["\']?\s*[:=]\s*["\']?[\w-]+', 'api_key=***', sanitized)
        return sanitized
    
    async def _execute_anthropic(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute with Anthropic API - REAL CALLS."""
        client = self.providers["anthropic"]
        
        # Check if we have messages (multi-turn) or prompt (single-turn)
        messages = request.parameters.get("messages")
        if not messages:
            # Single-turn: build messages from prompt
            messages = [{"role": "user", "content": request.prompt}]
        
        # Get parameters
        max_tokens = request.constraints.get("max_tokens", 4000)
        temperature = request.constraints.get("temperature", 0.7)
        
        # Make REAL API call
        response = await client.messages.create(
            model=request.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Calculate cost (pricing as of 2024)
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
            model_name=None,  # Level 2: Hide model name
            metadata={
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "blinded_metadata": True
            },
            execution_time=0,  # Will be set by caller
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            cost=total_cost
        )
    
    async def _execute_openai(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute with OpenAI API - REAL CALLS."""
        client = self.providers["openai"]
        
        # Check if we have messages (multi-turn) or prompt (single-turn)
        messages = request.parameters.get("messages")
        if not messages:
            # Single-turn: build messages from prompt
            messages = [{"role": "user", "content": request.prompt}]
        
        # Get parameters
        max_tokens = request.constraints.get("max_tokens", 4000)
        temperature = request.constraints.get("temperature", 0.7)
        
        # Make REAL API call
        response = await client.chat.completions.create(
            model=request.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Calculate cost (pricing as of 2024)
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
            model_name=None,  # Level 2: Hide model name
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "blinded_metadata": True
            },
            execution_time=0,  # Will be set by caller
            tokens_used=response.usage.total_tokens,
            cost=total_cost
        )
    
    async def _execute_google(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute with Google API - REAL CALLS."""
        import asyncio
        genai = self.providers["google"]
        
        # Google doesn't support messages format directly
        # We need to convert messages to a single prompt
        messages = request.parameters.get("messages")
        if messages:
            # Convert messages to single prompt
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                else:
                    prompt_parts.append(f"Assistant: {msg['content']}")
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
        else:
            prompt = request.prompt
        
        # Get parameters
        max_tokens = request.constraints.get("max_tokens", 4000)
        temperature = request.constraints.get("temperature", 0.7)
        
        # Create model
        model = genai.GenerativeModel(request.model_id)
        
        # Google's API is sync, so we run in executor
        def _generate():
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return response
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _generate)
        
        # Estimate tokens (Google doesn't always provide exact counts)
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(response.text.split()) * 1.3
        
        # Calculate cost (pricing as of 2024)
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
            model_name=None,  # Level 2: Hide model name
            metadata={
                "estimated_tokens": True,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "blinded_metadata": True
            },
            execution_time=0,  # Will be set by caller
            tokens_used=int(input_tokens + output_tokens),
            cost=total_cost
        )
    
    async def unblind(self, blinding_id: str) -> Dict[str, Any]:
        """Retrieve original model info from blinding ID."""
        filepath = os.path.join(self.blinding_dir, f"{blinding_id}.json")
        
        if not os.path.exists(filepath):
            raise ValueError(f"No blinding found for ID: {blinding_id}")
        
        async with aiofiles.open(filepath, 'r') as f:
            content = await f.read()
            return json.loads(content)
