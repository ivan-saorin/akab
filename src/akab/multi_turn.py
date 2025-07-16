"""
Multi-Turn Execution Support for AKAB
Enables continuation-based testing for fair model comparisons
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiTurnExecutor:
    """Handles multi-turn execution with continuation support"""
    
    def __init__(self, hermes, sampling_manager):
        self.hermes = hermes
        self.sampling_manager = sampling_manager
        
    async def execute_with_continuation(self, provider: str, model: str,
                                      initial_prompt: str, constraints: Dict[str, Any],
                                      max_turns: int = 10,
                                      target_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Execute a prompt with multi-turn continuation support"""
        
        logger.info(f"\nStarting multi-turn execution: {model} ({provider})")
        logger.info(f"Max turns: {max_turns}, Target tokens: {target_tokens}")
        
        full_response = []
        total_tokens = 0
        total_cost = 0
        turn_count = 0
        execution_times = []
        
        # Build conversation as standard messages format - works for ALL providers
        messages = []
        
        while turn_count < max_turns:
            turn_count += 1
            logger.info(f"\nTurn {turn_count}/{max_turns}")
            
            start_time = time.time()
            
            # First turn: add initial prompt
            if turn_count == 1:
                messages.append({"role": "user", "content": initial_prompt})
            else:
                # Continuation: simple and consistent
                messages.append({"role": "user", "content": "continue"})
            
            try:
                # Execute with full conversation context
                result = await self._execute_single_turn(
                    provider, model, messages, constraints
                )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # Extract response
                response_content = result.get("content", "")
                if not response_content:
                    logger.error(f"Empty response on turn {turn_count}")
                    break
                
                # Add response to conversation
                messages.append({"role": "assistant", "content": response_content})
                full_response.append(response_content)
                
                # Update metrics
                tokens_used = result.get("usage", {}).get("total_tokens", 0)
                total_tokens += tokens_used
                total_cost += result.get("cost", 0)
                
                # SIMPLE completion check: if LLM says [END], it's done!
                if "[END]" in response_content.upper():
                    logger.info(f"Found [END] marker after {turn_count} turns")
                    break
                
                # Check if we've reached target tokens
                if target_tokens and total_tokens >= target_tokens:
                    logger.info(f"Reached target tokens ({total_tokens}/{target_tokens})")
                    break
                    
            except Exception as e:
                logger.error(f"Error in turn {turn_count}: {str(e)}")
                break
        
        # Compile final result
        final_response = "\n".join(full_response)
        
        return {
            "content": final_response,
            "turns_used": turn_count,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "execution_times": execution_times,
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "provider": provider,
            "model": model,
            "completed_naturally": "[END]" in final_response.upper(),
            "response_length": len(final_response)
        }
    
    async def _execute_single_turn(self, provider: str, model: str,
                                  messages: List[Dict[str, str]], 
                                  constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single turn with conversation context"""
        
        from substrate import ExecutionRequest
        
        # Build proper request with messages format
        exec_request = ExecutionRequest(
            prompt=None,  # We're using messages instead
            model_id=model,
            model_name=model,
            parameters={
                "provider": provider,
                "messages": messages  # This is what was missing!
            },
            constraints=constraints
        )
        
        # Execute
        exec_result = await self.hermes.execute(exec_request)
        
        # Return standardized result
        return {
            "content": exec_result.response or "",
            "usage": {
                "total_tokens": exec_result.tokens_used or 0
            },
            "cost": exec_result.cost or 0,
            "stop_reason": exec_result.metadata.get("stop_reason") if exec_result.metadata else None
        }
    
    async def execute_campaign_with_continuation(self, campaign, iterations: int = 1,
                                               max_turns_per_test: int = 10,
                                               target_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute entire campaign with multi-turn support"""
        results = []
        
        for variant in campaign.variants:
            variant_id = variant["id"]
            
            # Simple check: does this need multi-turn?
            needs_multi_turn = variant.get("multi_turn", False) or variant.get("target_tokens", 0) > 4000
            
            for iteration in range(iterations):
                logger.info(f"\nExecuting {variant_id}, iteration {iteration + 1}/{iterations}")
                
                try:
                    if needs_multi_turn:
                        # Multi-turn execution
                        result = await self.execute_with_continuation(
                            provider=variant["provider"],
                            model=variant["model"],
                            initial_prompt=variant["prompt"],
                            constraints=variant.get("constraints", {}),
                            max_turns=max_turns_per_test,
                            target_tokens=variant.get("target_tokens", target_tokens)
                        )
                    else:
                        # Single-turn execution
                        messages = [{"role": "user", "content": variant["prompt"]}]
                        result = await self._execute_single_turn(
                            provider=variant["provider"],
                            model=variant["model"],
                            messages=messages,
                            constraints=variant.get("constraints", {})
                        )
                        result["turns_used"] = 1
                        result["response_length"] = len(result.get("content", ""))
                    
                    # Add metadata
                    result["variant"] = variant_id
                    result["iteration"] = iteration
                    result["success"] = True
                    result["timestamp"] = time.time()
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error executing variant {variant_id}: {str(e)}")
                    results.append({
                        "variant": variant_id,
                        "iteration": iteration,
                        "success": False,
                        "error": str(e),
                        "timestamp": time.time(),
                        "response_length": 0,  # Always include this!
                        "total_tokens": 0,
                        "turns_used": 0
                    })
        
        return results


class EnhancedCampaignExecutor:
    """Enhanced campaign executor with multi-turn and pattern support"""
    
    def __init__(self, multi_turn_executor: MultiTurnExecutor,
                 pattern_enhancer: Optional[Any] = None,
                 model_sizes: Optional[Dict[str, Dict[str, str]]] = None):
        self.multi_turn_executor = multi_turn_executor
        self.pattern_enhancer = pattern_enhancer
        self.model_sizes = model_sizes or {}
        
    async def create_enhanced_campaign(self, name: str, description: str,
                                     base_prompt: str, models: List[Dict[str, str]],
                                     enhancement_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create campaign with automatic enhancement variants"""
        
        variants = []
        
        for model_config in models:
            provider = model_config["provider"]
            
            # Get actual model name from size nickname
            if "size" in model_config and provider in self.model_sizes:
                model = self.model_sizes[provider].get(model_config["size"], model_config.get("model", ""))
            else:
                model = model_config.get("model", "")
            
            model_id = f"{provider}_{model_config.get('size', 'm')}"
            
            # Create variant config
            base_variant = {
                "provider": provider,
                "model": model,
                "prompt": base_prompt,
                "multi_turn": enhancement_config.get("multi_turn", False),
                "target_tokens": enhancement_config.get("target_tokens"),
                "constraints": {
                    "temperature": enhancement_config.get("temperature", 0.7),
                    "max_tokens": enhancement_config.get("max_tokens", 4000)
                }
            }
            
            # Baseline variant
            if enhancement_config.get("include_baseline", True):
                variants.append({
                    "id": f"{model_id}_baseline",
                    **base_variant
                })
            
            # Enhanced variant (if enhancer available)
            if self.pattern_enhancer and enhancement_config.get("enhance", True):
                enhanced_result = self.pattern_enhancer.enhance_prompt(
                    prompt=base_prompt,
                    model=model_id,
                    strategy=enhancement_config.get("strategy", "auto")
                )
                
                variants.append({
                    "id": f"{model_id}_enhanced",
                    **base_variant,
                    "prompt": enhanced_result["enhanced_prompt"],
                    "patterns_applied": enhanced_result.get("patterns_applied", [])
                })
        
        return {
            "name": name,
            "description": description,
            "variants": variants,
            "success_criteria": enhancement_config.get("success_criteria", {
                "primary_metric": "response_length",
                "evaluation_method": "statistical"
            })
        }
    
    def calculate_fair_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate fair comparison metrics accounting for multi-turn"""
        
        metrics_by_variant = {}
        
        for result in results:
            if not result.get("success", False):
                continue
                
            variant_id = result["variant"]
            if variant_id not in metrics_by_variant:
                metrics_by_variant[variant_id] = []
            
            metrics_by_variant[variant_id].append({
                "total_tokens": result.get("total_tokens", 0),
                "response_length": result.get("response_length", len(result.get("content", ""))),
                "turns_used": result.get("turns_used", 1),
                "cost": result.get("total_cost", result.get("cost", 0)),
                "execution_time": result.get("avg_execution_time", 0)
            })
        
        # Calculate aggregates
        fair_comparison = {}
        
        for variant_id, metrics_list in metrics_by_variant.items():
            if not metrics_list:
                continue
                
            fair_comparison[variant_id] = {
                "avg_total_tokens": sum(m["total_tokens"] for m in metrics_list) / len(metrics_list),
                "avg_response_length": sum(m["response_length"] for m in metrics_list) / len(metrics_list),
                "avg_turns_used": sum(m["turns_used"] for m in metrics_list) / len(metrics_list),
                "total_cost": sum(m["cost"] for m in metrics_list),
                "avg_cost_per_test": sum(m["cost"] for m in metrics_list) / len(metrics_list),
                "avg_execution_time": sum(m["execution_time"] for m in metrics_list) / len(metrics_list),
                "test_count": len(metrics_list)
            }
        
        return fair_comparison
