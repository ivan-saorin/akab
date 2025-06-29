"""Constraint suggestion system for AKAB."""

import logging
from typing import Any, Dict, List, Optional

# Import substrate components
try:
    from substrate import SamplingManager
except ImportError:
    try:
        from .substrate import SamplingManager
    except ImportError:
        from .substrate_stub import SamplingManager

logger = logging.getLogger(__name__)


class ConstraintSuggester:
    """Suggests constraints for A/B testing using sampling."""
    
    # Default constraint templates
    DEFAULT_CONSTRAINTS = {
        "general": {
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.95,
        },
        "creative": {
            "max_tokens": 2000,
            "temperature": 0.9,
            "top_p": 0.95,
        },
        "analytical": {
            "max_tokens": 1500,
            "temperature": 0.3,
            "top_p": 0.90,
        },
        "code": {
            "max_tokens": 2000,
            "temperature": 0.2,
            "top_p": 0.95,
        },
        "conversation": {
            "max_tokens": 500,
            "temperature": 0.8,
            "top_p": 0.95,
        }
    }
    
    def __init__(self, sampling_manager: SamplingManager):
        """Initialize constraint suggester.
        
        Args:
            sampling_manager: Sampling manager for LLM assistance
        """
        self.sampling_manager = sampling_manager
        
    def get_default_constraints(self, prompt_type: str = "general") -> Dict[str, Any]:
        """Get default constraints for a prompt type.
        
        Args:
            prompt_type: Type of prompt (general, creative, analytical, etc.)
            
        Returns:
            Default constraints
        """
        return self.DEFAULT_CONSTRAINTS.get(
            prompt_type,
            self.DEFAULT_CONSTRAINTS["general"]
        ).copy()
        
    def analyze_prompt(self, prompt: str) -> str:
        """Analyze prompt to determine its type.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Prompt type (general, creative, analytical, code, conversation)
        """
        prompt_lower = prompt.lower()
        
        # Simple heuristics
        if any(word in prompt_lower for word in ["code", "function", "implement", "debug", "program"]):
            return "code"
        elif any(word in prompt_lower for word in ["analyze", "compare", "evaluate", "assess", "examine"]):
            return "analytical"
        elif any(word in prompt_lower for word in ["create", "write", "story", "poem", "imagine"]):
            return "creative"
        elif any(word in prompt_lower for word in ["chat", "talk", "discuss", "conversation"]):
            return "conversation"
        else:
            return "general"
            
    def suggest_constraints(
        self,
        prompt: str,
        providers: List[str],
        existing_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Suggest constraints for a prompt and providers.
        
        Args:
            prompt: The prompt to test
            providers: List of providers
            existing_constraints: Any existing constraints to merge
            
        Returns:
            Suggested constraints
        """
        # Determine prompt type
        prompt_type = self.analyze_prompt(prompt)
        
        # Start with defaults
        constraints = self.get_default_constraints(prompt_type)
        
        # Adjust based on providers
        if any("xl" in p or "xxl" in p for p in providers):
            # Larger models can handle more tokens
            constraints["max_tokens"] = min(4000, constraints["max_tokens"] * 2)
            
        if any("xs" in p or "s" in p for p in providers):
            # Smaller models might need lower token limits
            constraints["max_tokens"] = min(1000, constraints["max_tokens"])
            
        # Merge with existing constraints
        if existing_constraints:
            constraints.update(existing_constraints)
            
        return constraints
        
    async def get_sampling_suggestions(
        self,
        prompt: str,
        providers: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get constraint suggestions via sampling.
        
        Args:
            prompt: The prompt to test
            providers: List of providers
            context: Optional context
            
        Returns:
            Sampling request dict if sampling is appropriate
        """
        if not self.sampling_manager.should_request_sampling("constraints"):
            return None
            
        sampling_prompt = f"""
        The user wants to A/B test this prompt across these providers: {providers}
        
        Prompt: "{prompt[:200]}..."
        
        Based on the prompt content and providers, what constraints would you suggest?
        Consider: max_tokens, temperature, top_p, and any other relevant parameters.
        
        Explain your reasoning briefly.
        """
        
        return self.sampling_manager.create_request(
            sampling_prompt,
            max_tokens=150,
            temperature=0.7,
            context=context
        )
        
    def validate_constraints(
        self,
        constraints: Dict[str, Any],
        providers: List[str]
    ) -> tuple[bool, List[str]]:
        """Validate constraints for providers.
        
        Args:
            constraints: Constraints to validate
            providers: List of providers
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check max_tokens
        max_tokens = constraints.get("max_tokens", 0)
        if max_tokens <= 0:
            issues.append("max_tokens must be positive")
        elif max_tokens > 4000:
            issues.append("max_tokens exceeds typical limits (4000)")
            
        # Check temperature
        temperature = constraints.get("temperature", 0.7)
        if not 0 <= temperature <= 2:
            issues.append("temperature should be between 0 and 2")
            
        # Check top_p
        top_p = constraints.get("top_p", 1.0)
        if not 0 < top_p <= 1:
            issues.append("top_p should be between 0 and 1")
            
        # Provider-specific checks
        for provider in providers:
            if "xs" in provider and max_tokens > 2000:
                issues.append(f"{provider} may not support {max_tokens} tokens")
                
        return len(issues) == 0, issues
        
    def merge_constraints(
        self,
        *constraint_sets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge multiple constraint sets intelligently.
        
        Args:
            *constraint_sets: Variable number of constraint dicts
            
        Returns:
            Merged constraints
        """
        merged = {}
        
        for constraints in constraint_sets:
            if not constraints:
                continue
                
            for key, value in constraints.items():
                if key not in merged:
                    merged[key] = value
                elif key == "max_tokens":
                    # Take the minimum for safety
                    merged[key] = min(merged[key], value)
                elif key in ["temperature", "top_p"]:
                    # Average these values
                    merged[key] = (merged[key] + value) / 2
                else:
                    # For other keys, last one wins
                    merged[key] = value
                    
        return merged
