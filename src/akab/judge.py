"""LLM Judge for quality evaluation in AKAB experiments"""

import json
from typing import List, Dict, Any, Optional
from substrate import ExecutionRequest


class QualityJudge:
    """Use LLM to evaluate response quality for Level 3 experiments"""
    
    def __init__(self, hermes_executor):
        """Initialize with a Hermes executor for LLM calls"""
        self.hermes = hermes_executor
        
    async def evaluate_responses(self, responses: List[Dict[str, Any]], 
                               prompt: str, 
                               criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses for quality using LLM judge.
        
        Args:
            responses: List of response dicts with 'variant', 'response', and metadata
            prompt: Original prompt that generated these responses
            criteria: Custom evaluation criteria
            
        Returns:
            List of evaluation results with quality scores
        """
        if not criteria:
            criteria = {
                "relevance": "How well does the response address the prompt?",
                "clarity": "How clear and well-structured is the response?",
                "completeness": "How thorough and complete is the response?",
                "accuracy": "How factually accurate is the response?",
                "helpfulness": "How helpful would this response be to a user?"
            }
        
        # Create evaluation prompt
        eval_prompt = self._create_evaluation_prompt(prompt, responses, criteria)
        
        # Use Claude to evaluate
        request = ExecutionRequest(
            prompt=eval_prompt,
            model_id="claude-3-5-sonnet-20241022",
            model_name="claude-3-5-sonnet-20241022",
            parameters={
                "provider": "anthropic",
                "purpose": "quality_evaluation"
            },
            constraints={
                "temperature": 0.0,  # Consistent evaluation
                "max_tokens": 2000
            }
        )
        
        result = await self.hermes.execute(request)
        
        if result.error:
            raise Exception(f"Quality evaluation failed: {result.error}")
        
        # Parse evaluation results
        try:
            evaluations = self._parse_evaluation_response(result.response)
            return evaluations
        except Exception as e:
            raise Exception(f"Failed to parse evaluation response: {e}")
    
    def _create_evaluation_prompt(self, prompt: str, responses: List[Dict[str, Any]], 
                                criteria: Dict[str, Any]) -> str:
        """Create the evaluation prompt for the LLM judge"""
        criteria_text = "\n".join([f"- {k}: {v}" for k, v in criteria.items()])
        
        # Format responses for evaluation
        response_text = ""
        for i, resp in enumerate(responses):
            variant_id = resp.get("variant", f"response_{i}")
            response_content = resp.get("response", "")
            response_text += f"\n\n### Response {variant_id}:\n{response_content}"
        
        eval_prompt = f"""You are an expert evaluator tasked with assessing the quality of AI responses.

Original Prompt:
{prompt}

Evaluation Criteria:
{criteria_text}

Please evaluate each response below on a scale of 0-100 for each criterion, and provide an overall quality score.
Be objective and consistent in your evaluation.

Responses to evaluate:
{response_text}

Please respond with a JSON array where each element has this structure:
{{
    "variant": "the variant ID",
    "scores": {{
        "relevance": 85,
        "clarity": 90,
        "completeness": 80,
        "accuracy": 95,
        "helpfulness": 88
    }},
    "overall_score": 88,
    "strengths": ["list of key strengths"],
    "weaknesses": ["list of key weaknesses"]
}}

IMPORTANT: Return ONLY the JSON array, no other text."""
        
        return eval_prompt
    
    def _parse_evaluation_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the JSON evaluation response"""
        # Try to extract JSON from the response
        response = response.strip()
        
        # Find JSON array in response
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON array found in response")
        
        json_str = response[start_idx:end_idx]
        
        try:
            evaluations = json.loads(json_str)
            return evaluations
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")
    
    async def evaluate_campaign_results(self, campaign_results: List[Dict[str, Any]], 
                                      prompts: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate all results from a campaign, grouped by variant.
        
        Returns:
            Dict mapping variant IDs to lists of quality scores
        """
        quality_scores_by_variant = {}
        
        # Group results by prompt for batch evaluation
        for prompt_idx, prompt in enumerate(prompts):
            # Get all responses for this prompt
            prompt_results = [r for r in campaign_results if r.get("prompt_index") == prompt_idx]
            
            if not prompt_results:
                continue
            
            # Prepare responses for evaluation
            responses_to_eval = []
            for result in prompt_results:
                if result.get("success") and result.get("response"):
                    responses_to_eval.append({
                        "variant": result["variant"],
                        "response": result["response"]
                    })
            
            if responses_to_eval:
                # Evaluate this batch
                evaluations = await self.evaluate_responses(responses_to_eval, prompt)
                
                # Store scores by variant
                for eval_result in evaluations:
                    variant_id = eval_result["variant"]
                    score = eval_result["overall_score"]
                    
                    if variant_id not in quality_scores_by_variant:
                        quality_scores_by_variant[variant_id] = []
                    
                    quality_scores_by_variant[variant_id].append(score)
        
        return quality_scores_by_variant
