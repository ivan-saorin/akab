"""Laboratory - Scientific controls for A/B testing"""

import random
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import stats


class Laboratory:
    """
    Provides scientific controls for A/B testing:
    - Blinding service
    - Randomization
    - Statistical engine (trimmed means)
    - Result validation
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility"""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    # Blinding Service
    def create_blind_mapping(self, variants: List[str], campaign_id: str) -> Dict[str, str]:
        """Create blinded IDs for variants"""
        mapping = {}
        
        for variant in variants:
            # Create deterministic blind ID
            hash_input = f"{campaign_id}|{variant}|{self.seed or 'noseed'}"
            hash_obj = hashlib.sha256(hash_input.encode())
            blind_id = f"variant_{hash_obj.hexdigest()[:6]}"
            mapping[variant] = blind_id
        
        return mapping
    
    def unblind_results(self, results: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """Unblind results using mapping"""
        reverse_mapping = {v: k for k, v in mapping.items()}
        unblinded = {}
        
        for blind_id, data in results.items():
            original_id = reverse_mapping.get(blind_id, blind_id)
            unblinded[original_id] = data
        
        return unblinded
    
    # Randomization
    def randomize_order(self, items: List[Any]) -> List[Any]:
        """Randomize order of items"""
        shuffled = items.copy()
        random.shuffle(shuffled)
        return shuffled
    
    def assign_to_groups(self, items: List[Any], n_groups: int) -> List[List[Any]]:
        """Randomly assign items to n groups"""
        shuffled = self.randomize_order(items)
        groups = [[] for _ in range(n_groups)]
        
        for i, item in enumerate(shuffled):
            group_idx = i % n_groups
            groups[group_idx].append(item)
        
        return groups
    
    # Statistical Engine
    def calculate_trimmed_mean(self, values: List[float], trim_percent: float = 0.1) -> float:
        """Calculate trimmed mean (removes outliers)"""
        if not values:
            return 0.0
        
        return float(stats.trim_mean(values, trim_percent))
    
    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics"""
        if not values:
            return {
                "mean": 0.0,
                "trimmed_mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0
            }
        
        arr = np.array(values)
        
        return {
            "mean": float(np.mean(arr)),
            "trimmed_mean": self.calculate_trimmed_mean(values),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(values)
        }
    
    def compare_groups(self, group_a: List[float], group_b: List[float]) -> Dict[str, Any]:
        """Compare two groups statistically"""
        stats_a = self.calculate_statistics(group_a)
        stats_b = self.calculate_statistics(group_b)
        
        # Perform t-test if enough samples
        if len(group_a) >= 2 and len(group_b) >= 2:
            t_stat, p_value = stats.ttest_ind(group_a, group_b)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(group_a)**2 + np.std(group_b)**2) / 2)
            effect_size = (stats_a["mean"] - stats_b["mean"]) / pooled_std if pooled_std > 0 else 0
        else:
            t_stat, p_value, effect_size = None, None, None
        
        return {
            "group_a": stats_a,
            "group_b": stats_b,
            "comparison": {
                "mean_difference": stats_a["mean"] - stats_b["mean"],
                "trimmed_mean_difference": stats_a["trimmed_mean"] - stats_b["trimmed_mean"],
                "t_statistic": float(t_stat) if t_stat is not None else None,
                "p_value": float(p_value) if p_value is not None else None,
                "effect_size": float(effect_size) if effect_size is not None else None,
                "significant": p_value < 0.05 if p_value is not None else None
            }
        }
    
    # Result Validation
    def validate_results(self, results: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate results for statistical soundness"""
        issues = []
        
        # Check for minimum sample size
        if len(results) < 10:
            issues.append(f"Low sample size: {len(results)} < 10")
        
        # Check for data completeness
        required_fields = ["variant", "score", "timestamp"]
        for i, result in enumerate(results):
            for field in required_fields:
                if field not in result:
                    issues.append(f"Result {i} missing field: {field}")
        
        # Check for score validity
        scores = [r.get("score", 0) for r in results]
        if scores:
            if all(s == scores[0] for s in scores):
                issues.append("All scores are identical - no variance")
            
            if any(not isinstance(s, (int, float)) for s in scores):
                issues.append("Non-numeric scores detected")
        
        # Check for temporal distribution
        if results:
            timestamps = sorted([r.get("timestamp", 0) for r in results])
            duration = timestamps[-1] - timestamps[0]
            
            if duration < 60:  # Less than 1 minute
                issues.append("Results collected too quickly - may lack diversity")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def generate_report(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical report"""
        results = campaign_data.get("results", [])
        
        # Group by variant
        variant_groups = {}
        for result in results:
            variant = result.get("variant", "unknown")
            if variant not in variant_groups:
                variant_groups[variant] = []
            variant_groups[variant].append(result.get("score", 0))
        
        # Calculate statistics for each variant
        variant_stats = {}
        for variant, scores in variant_groups.items():
            variant_stats[variant] = self.calculate_statistics(scores)
        
        # Pairwise comparisons
        comparisons = {}
        variants = list(variant_groups.keys())
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                key = f"{variants[i]}_vs_{variants[j]}"
                comparisons[key] = self.compare_groups(
                    variant_groups[variants[i]],
                    variant_groups[variants[j]]
                )
        
        # Validation
        is_valid, issues = self.validate_results(results)
        
        return {
            "campaign_id": campaign_data.get("id"),
            "total_results": len(results),
            "variant_statistics": variant_stats,
            "comparisons": comparisons,
            "validation": {
                "is_valid": is_valid,
                "issues": issues
            },
            "recommendation": self._generate_recommendation(variant_stats, comparisons)
        }
    
    def _generate_recommendation(self, variant_stats: Dict[str, Any], 
                               comparisons: Dict[str, Any]) -> str:
        """Generate recommendation based on analysis"""
        if not variant_stats:
            return "No data to analyze"
        
        # Find best performer by trimmed mean
        best_variant = max(variant_stats.items(), 
                          key=lambda x: x[1]["trimmed_mean"])
        
        # Check if statistically significant
        significant_wins = []
        for comp_key, comp_data in comparisons.items():
            if comp_data["comparison"]["significant"]:
                if comp_data["comparison"]["trimmed_mean_difference"] > 0:
                    # First variant wins
                    winner = comp_key.split("_vs_")[0]
                else:
                    # Second variant wins
                    winner = comp_key.split("_vs_")[1]
                significant_wins.append(winner)
        
        if significant_wins:
            # Count wins
            win_counts = {}
            for winner in significant_wins:
                win_counts[winner] = win_counts.get(winner, 0) + 1
            
            # Most wins
            champion = max(win_counts.items(), key=lambda x: x[1])
            return f"Variant '{champion[0]}' shows statistically significant improvement"
        else:
            return f"Variant '{best_variant[0]}' performs best but not statistically significant"
    
    def perform_t_test(self, group1: List[float], group2: List[float], 
                      significance_level: float = 0.05) -> Dict[str, Any]:
        """Perform two-sample t-test"""
        import math
        
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return {
                "p_value": 1.0,
                "significant": False,
                "error": "Insufficient data for t-test"
            }
        
        # Calculate means and variances
        mean1 = sum(group1) / n1
        mean2 = sum(group2) / n2
        
        var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
        
        # Pooled standard error
        se = math.sqrt(var1/n1 + var2/n2)
        
        if se == 0:
            return {
                "p_value": 1.0,
                "significant": False,
                "error": "Zero variance"
            }
        
        # T-statistic
        t_stat = (mean1 - mean2) / se
        
        # Degrees of freedom (Welch's t-test)
        df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Approximate p-value using normal distribution
        # (For production, use scipy.stats.t.cdf)
        from math import erf
        z = abs(t_stat) / math.sqrt(2)
        p_value = 2 * (1 - 0.5 * (1 + erf(z / math.sqrt(2))))
        
        return {
            "t_statistic": t_stat,
            "degrees_of_freedom": df,
            "p_value": p_value,
            "significant": p_value < significance_level,
            "mean_difference": mean1 - mean2,
            "confidence_level": 1 - significance_level
        }
    
    def calculate_effect_size(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return {"cohens_d": 0, "interpretation": "insufficient data"}
        
        mean1 = sum(group1) / n1
        mean2 = sum(group2) / n2
        
        # Pooled standard deviation
        var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
        pooled_std = ((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2)) ** 0.5
        
        if pooled_std == 0:
            return {"cohens_d": 0, "interpretation": "zero variance"}
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            "cohens_d": cohens_d,
            "interpretation": interpretation,
            "absolute_value": abs_d
        }
    
    def check_experiment_significance(self, variant_results: Dict[str, List[float]], 
                                    significance_level: float = 0.05,
                                    effect_size_threshold: float = 0.2) -> Dict[str, Any]:
        """Check if experiment has reached statistical significance"""
        variants = list(variant_results.keys())
        if len(variants) < 2:
            return {
                "significant": False,
                "reason": "Need at least 2 variants"
            }
        
        # Perform pairwise comparisons
        comparisons = []
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                v1, v2 = variants[i], variants[j]
                scores1 = variant_results[v1]
                scores2 = variant_results[v2]
                
                # T-test
                t_test = self.perform_t_test(scores1, scores2, significance_level)
                
                # Effect size
                effect = self.calculate_effect_size(scores1, scores2)
                
                comparisons.append({
                    "variant1": v1,
                    "variant2": v2,
                    "t_test": t_test,
                    "effect_size": effect,
                    "meaningful": t_test["significant"] and abs(effect["cohens_d"]) >= effect_size_threshold
                })
        
        # Check if any comparison is meaningful
        meaningful_comparisons = [c for c in comparisons if c["meaningful"]]
        
        if meaningful_comparisons:
            # Find the best performer
            variant_means = {v: sum(scores)/len(scores) for v, scores in variant_results.items()}
            best_variant = max(variant_means.items(), key=lambda x: x[1])[0]
            
            return {
                "significant": True,
                "best_variant": best_variant,
                "comparisons": comparisons,
                "meaningful_comparisons": len(meaningful_comparisons)
            }
        else:
            return {
                "significant": False,
                "reason": "No meaningful differences found",
                "comparisons": comparisons
            }
