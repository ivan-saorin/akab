"""Laboratory Executor - AKAB's campaign execution engine"""
import random
import hashlib
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Laboratory:
    """
    Provides scientific controls for A/B testing:
    - Blinding service
    - Randomization
    - Statistical engine (trimmed means)
    - Result validation
    - Significance testing
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
            # Create deterministic blind ID based on campaign and variant
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
                "std_dev": 0.0,  # Alias for compatibility
                "min": 0.0,
                "max": 0.0,
                "count": 0
            }
        
        arr = np.array(values)
        std_val = float(np.std(arr))
        
        return {
            "mean": float(np.mean(arr)),
            "trimmed_mean": self.calculate_trimmed_mean(values),
            "median": float(np.median(arr)),
            "std": std_val,
            "std_dev": std_val,  # Alias for compatibility
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(values)
        }
    
    def perform_hypothesis_test(
        self, 
        group1: List[float], 
        group2: List[float],
        test_type: str = "welch"
    ) -> Dict[str, Any]:
        """Perform hypothesis test between two groups
        
        Args:
            group1: First group of values
            group2: Second group of values
            test_type: Type of test ("welch" for Welch's t-test, "mann_whitney" for non-parametric)
            
        Returns:
            Dict with test results including p-value and effect size
        """
        if not group1 or not group2:
            return {
                "test": test_type,
                "p_value": 1.0,
                "statistic": 0.0,
                "significant": False,
                "effect_size": {"cohens_d": 0.0}
            }
        
        arr1 = np.array(group1)
        arr2 = np.array(group2)
        
        # Perform test
        if test_type == "welch":
            statistic, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)
        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(arr1) + np.var(arr2)) / 2)
        cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            "test": test_type,
            "p_value": float(p_value),
            "statistic": float(statistic),
            "significant": p_value < 0.05,
            "effect_size": {
                "cohens_d": float(cohens_d),
                "interpretation": self._interpret_cohens_d(abs(cohens_d))
            }
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def check_experiment_significance(
        self,
        variant_results: Dict[str, List[float]],
        significance_level: float = 0.05,
        effect_size_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """Check if experiment has reached statistical significance
        
        Args:
            variant_results: Dict mapping variant IDs to lists of values
            significance_level: P-value threshold for significance
            effect_size_threshold: Minimum Cohen's d for practical significance
            
        Returns:
            Dict with significance results and best variant
        """
        if len(variant_results) < 2:
            return {
                "significant": False,
                "reason": "Need at least 2 variants with results",
                "comparisons": []
            }
        
        # Calculate statistics for each variant
        variant_stats = {}
        for variant_id, values in variant_results.items():
            if values:
                variant_stats[variant_id] = self.calculate_statistics(values)
        
        if len(variant_stats) < 2:
            return {
                "significant": False,
                "reason": "Need at least 2 variants with data",
                "comparisons": []
            }
        
        # Perform pairwise comparisons
        comparisons = []
        variant_ids = list(variant_stats.keys())
        
        for i in range(len(variant_ids)):
            for j in range(i + 1, len(variant_ids)):
                v1, v2 = variant_ids[i], variant_ids[j]
                
                # Perform hypothesis test
                test_result = self.perform_hypothesis_test(
                    variant_results[v1],
                    variant_results[v2]
                )
                
                comparisons.append({
                    "variant1": v1,
                    "variant2": v2,
                    "p_value": test_result["p_value"],
                    "effect_size": test_result["effect_size"],
                    "significant": test_result["p_value"] < significance_level,
                    "practical_significance": abs(test_result["effect_size"]["cohens_d"]) >= effect_size_threshold
                })
        
        # Check if any comparison is both statistically and practically significant
        significant_comparisons = [
            c for c in comparisons 
            if c["significant"] and c["practical_significance"]
        ]
        
        if not significant_comparisons:
            # Check why not significant
            if all(c["significant"] for c in comparisons):
                reason = f"Effect sizes too small (< {effect_size_threshold})"
            elif any(c["practical_significance"] for c in comparisons):
                reason = f"P-values too high (> {significance_level})"
            else:
                reason = "No significant differences found"
            
            return {
                "significant": False,
                "reason": reason,
                "comparisons": comparisons
            }
        
        # Find best variant (lowest mean for metrics to minimize)
        best_variant = min(
            variant_stats.items(),
            key=lambda x: x[1]["trimmed_mean"]
        )[0]
        
        return {
            "significant": True,
            "best_variant": best_variant,
            "variant_stats": variant_stats,
            "comparisons": comparisons,
            "significant_comparisons": significant_comparisons
        }
    
    # Result Validation
    def validate_result(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a single test result
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["variant", "execution_time", "success"]
        
        for field in required_fields:
            if field not in result:
                return False, f"Missing required field: {field}"
        
        if not result["success"]:
            return True, None  # Failed results are valid
        
        # Validate execution time
        if result["execution_time"] < 0:
            return False, "Negative execution time"
        
        if result["execution_time"] > 300:  # 5 minutes
            return False, "Execution time exceeds reasonable limit"
        
        return True, None
    
    def validate_campaign_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate all campaign results
        
        Returns:
            Dict with validation summary
        """
        total = len(results)
        valid = 0
        invalid = 0
        errors = []
        
        for result in results:
            is_valid, error = self.validate_result(result)
            if is_valid:
                valid += 1
            else:
                invalid += 1
                errors.append({
                    "result": result.get("variant", "unknown"),
                    "error": error
                })
        
        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "errors": errors[:10]  # Limit error list
        }


# Global laboratory instance
LABORATORY = Laboratory()
