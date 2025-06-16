"""
AKAB Evaluation Engine
Handles scoring, metrics, and analysis of experiment results
"""
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Handles evaluation and scoring of experiment results"""
    
    # Keywords for innovation scoring
    INNOVATION_KEYWORDS = {
        "high": [
            "revolutionary", "breakthrough", "paradigm", "novel", "unprecedented",
            "cutting-edge", "innovative", "disruptive", "transformative", "quantum"
        ],
        "medium": [
            "creative", "unique", "original", "fresh", "inventive", "imaginative",
            "unconventional", "experimental", "progressive", "modern"
        ],
        "low": [
            "traditional", "conventional", "standard", "typical", "common",
            "established", "routine", "ordinary", "basic", "simple"
        ]
    }
    
    # Keywords for BS detection
    BS_KEYWORDS = [
        "quantum", "revolutionary", "game-changing", "disruptive",
        "paradigm shift", "breakthrough", "cutting-edge", "next-gen",
        "artificial general intelligence", "singularity", "exponential"
    ]
    
    def __init__(self):
        self.metrics = {
            "innovation_score": self.calculate_innovation_score,
            "coherence_score": self.calculate_coherence_score,
            "practicality_score": self.calculate_practicality_score,
            "bs_count": self.calculate_bs_count,
            "key_concepts": self.extract_key_concepts
        }
    
    async def evaluate_response(
        self,
        response: str,
        experiment_config: Dict[str, Any],
        provider: str
    ) -> Dict[str, Any]:
        """Evaluate a single response"""
        evaluation = {
            "provider": provider,
            "timestamp": datetime.now().isoformat(),
            "response_length": len(response.split()),
            "metrics": {}
        }
        
        # Calculate all metrics
        for metric_name, metric_func in self.metrics.items():
            try:
                evaluation["metrics"][metric_name] = metric_func(response)
            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {e}")
                evaluation["metrics"][metric_name] = None
        
        # Add composite score
        evaluation["composite_score"] = self._calculate_composite_score(
            evaluation["metrics"]
        )
        
        return evaluation
    
    def calculate_innovation_score(self, response: str) -> float:
        """Calculate innovation score (0-10)"""
        response_lower = response.lower()
        
        high_count = sum(1 for kw in self.INNOVATION_KEYWORDS["high"] 
                        if kw in response_lower)
        medium_count = sum(1 for kw in self.INNOVATION_KEYWORDS["medium"] 
                          if kw in response_lower)
        low_count = sum(1 for kw in self.INNOVATION_KEYWORDS["low"] 
                       if kw in response_lower)
        
        # Base score
        score = 5.0
        
        # Adjust based on keyword presence
        score += min(high_count * 0.5, 2.5)
        score += min(medium_count * 0.3, 1.5)
        score -= min(low_count * 0.2, 1.0)
        
        # Check for specific patterns
        if "novel approach" in response_lower:
            score += 0.5
        if "traditional method" in response_lower:
            score -= 0.5
        
        # Bound between 0 and 10
        return max(0, min(10, round(score, 1)))
    
    def calculate_coherence_score(self, response: str) -> float:
        """Calculate coherence score (0-10)"""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 5.0
        
        # Base score
        score = 7.0
        
        # Check for structure
        if any(s.startswith(('First', 'Second', 'Finally', 'In conclusion')) 
               for s in sentences):
            score += 1.0
        
        # Check for transitions
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 
                          'additionally', 'consequently']
        transition_count = sum(1 for word in transition_words 
                             if word in response.lower())
        score += min(transition_count * 0.3, 1.0)
        
        # Check for consistency in length
        lengths = [len(s.split()) for s in sentences]
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            if variance < 100:  # Low variance is good
                score += 0.5
        
        return max(0, min(10, round(score, 1)))
    
    def calculate_practicality_score(self, response: str) -> float:
        """Calculate practicality score (0-10)"""
        response_lower = response.lower()
        
        # Practical indicators
        practical_keywords = [
            'step', 'implement', 'process', 'method', 'approach',
            'solution', 'framework', 'system', 'tool', 'technique'
        ]
        
        # Impractical indicators
        impractical_keywords = [
            'theoretical', 'abstract', 'conceptual', 'hypothetical',
            'speculative', 'futuristic', 'impossible', 'unrealistic'
        ]
        
        score = 6.0
        
        # Count practical keywords
        practical_count = sum(1 for kw in practical_keywords 
                            if kw in response_lower)
        score += min(practical_count * 0.4, 2.0)
        
        # Count impractical keywords
        impractical_count = sum(1 for kw in impractical_keywords 
                              if kw in response_lower)
        score -= min(impractical_count * 0.5, 2.0)
        
        # Check for specific patterns
        if re.search(r'\d+\.?\s*(step|phase|stage)', response_lower):
            score += 1.0  # Numbered steps
        
        if 'example' in response_lower or 'case study' in response_lower:
            score += 0.5
        
        return max(0, min(10, round(score, 1)))
    
    def calculate_bs_count(self, response: str) -> int:
        """Count BS/buzzword occurrences"""
        response_lower = response.lower()
        
        count = 0
        for keyword in self.BS_KEYWORDS:
            # Count occurrences, but avoid double counting overlapping terms
            count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', 
                                   response_lower))
        
        return count
    
    def extract_key_concepts(self, response: str) -> List[str]:
        """Extract key concepts from response"""
        # Simple extraction based on capitalized phrases and key patterns
        concepts = []
        
        # Find capitalized phrases (potential concept names)
        cap_pattern = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
        capitalized = re.findall(cap_pattern, response)
        
        # Filter out common words and short phrases
        common_words = {'The', 'This', 'That', 'These', 'Those', 'Here', 
                       'There', 'What', 'When', 'Where', 'Why', 'How'}
        
        for phrase in capitalized:
            if (len(phrase.split()) >= 2 and 
                phrase.split()[0] not in common_words and
                len(phrase) > 10):
                concepts.append(phrase)
        
        # Find quoted concepts
        quoted = re.findall(r'"([^"]+)"', response)
        concepts.extend([q for q in quoted if 10 < len(q) < 50])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept.lower() not in seen:
                seen.add(concept.lower())
                unique_concepts.append(concept)
        
        return unique_concepts[:10]  # Limit to top 10
    
    def _calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite score from individual metrics"""
        weights = {
            "innovation_score": 0.3,
            "coherence_score": 0.3,
            "practicality_score": 0.3,
            "bs_penalty": 0.1
        }
        
        score = 0.0
        
        # Add weighted scores
        if metrics.get("innovation_score") is not None:
            score += metrics["innovation_score"] * weights["innovation_score"]
        
        if metrics.get("coherence_score") is not None:
            score += metrics["coherence_score"] * weights["coherence_score"]
        
        if metrics.get("practicality_score") is not None:
            score += metrics["practicality_score"] * weights["practicality_score"]
        
        # Subtract BS penalty
        if metrics.get("bs_count") is not None:
            bs_penalty = min(metrics["bs_count"] * 0.5, 3.0)
            score -= bs_penalty * weights["bs_penalty"]
        
        return max(0, min(10, round(score, 1)))
    
    async def analyze_campaign_results(
        self,
        experiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze results from a full campaign"""
        if not experiments:
            return {
                "status": "error",
                "message": "No experiments to analyze"
            }
        
        analysis = {
            "campaign_name": experiments[0].get("campaign_id", "Unknown"),
            "analysis_date": datetime.now().isoformat(),
            "total_experiments": len(experiments),
            "provider_metrics": defaultdict(lambda: defaultdict(list)),
            "overall_metrics": defaultdict(list),
            "key_findings": [],
            "recommendations": []
        }
        
        # Aggregate metrics by provider
        for exp in experiments:
            if "result" not in exp or "evaluation" not in exp["result"]:
                continue
            
            provider = exp["result"]["evaluation"].get("provider", "unknown")
            metrics = exp["result"]["evaluation"].get("metrics", {})
            
            for metric, value in metrics.items():
                if value is not None:
                    analysis["provider_metrics"][provider][metric].append(value)
                    analysis["overall_metrics"][metric].append(value)
        
        # Calculate averages
        for provider, metrics in analysis["provider_metrics"].items():
            for metric, values in metrics.items():
                if values:
                    avg = sum(values) / len(values)
                    analysis["provider_metrics"][provider][f"{metric}_avg"] = round(avg, 2)
        
        # Overall averages
        for metric, values in analysis["overall_metrics"].items():
            if values:
                avg = sum(values) / len(values)
                analysis["overall_metrics"][f"{metric}_avg"] = round(avg, 2)
        
        # Generate insights
        analysis["key_findings"] = self._generate_insights(analysis)
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        # Best performing provider
        if analysis["provider_metrics"]:
            best_provider = max(
                analysis["provider_metrics"].items(),
                key=lambda x: x[1].get("composite_score_avg", 0)
            )
            analysis["best_provider"] = {
                "name": best_provider[0],
                "composite_score": best_provider[1].get("composite_score_avg", 0)
            }
        
        return analysis
    
    def _generate_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate key insights from analysis"""
        insights = []
        
        # Innovation insights
        if "innovation_score_avg" in analysis["overall_metrics"]:
            avg_innovation = analysis["overall_metrics"]["innovation_score_avg"]
            if avg_innovation > 7:
                insights.append(f"High innovation across all providers (avg: {avg_innovation})")
            elif avg_innovation < 5:
                insights.append(f"Low innovation scores suggest conservative responses (avg: {avg_innovation})")
        
        # Provider comparisons
        provider_scores = {
            p: m.get("composite_score_avg", 0) 
            for p, m in analysis["provider_metrics"].items()
        }
        
        if len(provider_scores) > 1:
            best = max(provider_scores.items(), key=lambda x: x[1])
            worst = min(provider_scores.items(), key=lambda x: x[1])
            
            if best[1] - worst[1] > 2:
                insights.append(
                    f"Significant performance gap: {best[0]} ({best[1]}) vs {worst[0]} ({worst[1]})"
                )
        
        # BS detection
        bs_averages = {
            p: m.get("bs_count_avg", 0)
            for p, m in analysis["provider_metrics"].items()
        }
        
        high_bs = [p for p, bs in bs_averages.items() if bs > 3]
        if high_bs:
            insights.append(f"High buzzword usage detected in: {', '.join(high_bs)}")
        
        return insights
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Cost vs performance
        provider_metrics = analysis.get("provider_metrics", {})
        
        # Find best value (high score, low cost)
        # This is simplified - in reality, we'd need cost data
        if provider_metrics:
            scores = {
                p: m.get("composite_score_avg", 0)
                for p, m in provider_metrics.items()
            }
            
            if scores:
                best_score = max(scores.values())
                value_providers = [
                    p for p, s in scores.items() 
                    if s >= best_score * 0.9  # Within 90% of best
                ]
                
                if len(value_providers) > 1:
                    recommendations.append(
                        f"Consider these providers for best value: {', '.join(value_providers)}"
                    )
        
        # Innovation vs practicality
        innovation_avg = analysis["overall_metrics"].get("innovation_score_avg", 0)
        practicality_avg = analysis["overall_metrics"].get("practicality_score_avg", 0)
        
        if innovation_avg > 7 and practicality_avg < 5:
            recommendations.append(
                "High innovation but low practicality - consider balancing creativity with implementation details"
            )
        elif innovation_avg < 5 and practicality_avg > 7:
            recommendations.append(
                "High practicality but low innovation - consider encouraging more creative approaches"
            )
        
        # Scale recommendations
        total_exp = analysis.get("total_experiments", 0)
        if total_exp < 10:
            recommendations.append(
                "Consider running more experiments (20+) for statistically significant results"
            )
        
        return recommendations
