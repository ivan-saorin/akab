# AKAB Scientific Methodology

## Overview

AKAB implements a rigorous scientific framework for conducting unbiased experiments on Large Language Models (LLMs). The system provides three levels of testing rigor - from quick comparisons to fully-blinded scientific experiments. At its highest level, AKAB employs triple-blinding methodology with complete isolation - the entire /krill/ data directory exists outside LLM-accessible paths, ensuring perfect experimental integrity.

## The Problem: Cognitive Bias in LLM Evaluation

When evaluating LLMs, researchers face unique challenges:

1. **Confirmation Bias**: Favoring results that confirm pre-existing beliefs about certain models
2. **Anchoring Bias**: Being influenced by prior experiences with specific models
3. **Halo Effect**: Allowing positive impressions of a company/model to influence objective evaluation
4. **Attribution Bias**: Unconsciously attributing response characteristics to known models

Even well-intentioned researchers cannot fully eliminate these biases through willpower alone. The knowledge of which model produced which response inevitably influences interpretation.

## The Solution: Triple-Blind Experimental Design with Complete Isolation

### Complete Workflow Overview

1. **Design**: Create campaigns/experiments with appropriate level of rigor
2. **Execute**: Run tests with automatic blinding based on level
3. **Analyze**: Review results while maintaining blinding
4. **Reveal** (L3 only): Check statistical significance and reveal winners
5. **Unlock**: Archive completed work and access full mappings

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│           LLM Access (via MCP only)         │
│                                             │
│  ✓ AKAB MCP tools (quick, campaign, exp)   │
│  ✗ /krill/* (COMPLETELY INACCESSIBLE)      │
│  ✗ Direct file access to experiments        │
│  ✗ Direct file access to mappings          │
│  ✗ Any way to peek at model identities     │
└─────────────────────────────────────────────┘
                      ║
              Complete Isolation
                      ║
┌─────────────────────────────────────────────┐
│         AKAB MCP (Python Runtime)           │
│                                             │
│  ✓ Full access to /krill/                  │
│  ✓ Can read/write all test data            │
│  ✓ Can access scrambling mappings          │
│  ✓ Makes real API calls with real models   │
└─────────────────────────────────────────────┘
```

### Three Levels of Testing Rigor

AKAB provides three levels of testing, each with appropriate blinding:

#### Level 1 - Quick Compare (No Blinding)

- **Purpose**: Debugging, exploration, improvement testing
- **Blinding**: None - full visibility for rapid iteration
- **Winner Selection**: Human judgment only
- **Use Case**: "Does this prompt work better than that one?"

#### Level 2 - Campaign (Execution Blinding)

- **Purpose**: Standard A/B testing with debug capability
- **Blinding**: During execution, unlockable after completion
- **Winner Selection**: Automated based on success criteria
- **Use Case**: "Which approach produces more concise responses?"

#### Level 3 - Experiment (Triple Blinding)

- **Purpose**: Unbiased scientific evaluation
- **Blinding**: Complete - scrambled from startup
- **Winner Selection**: Statistical significance required
- **Use Case**: "Which model should we use in production?"

### Why This Architecture is Unbreakable

Since the entire `/krill/` directory is outside `allowed_directories`, the LLM cannot:

- Read experiment files directly
- Access mapping files
- Browse the directory structure
- Find any way to correlate scrambled names to real models

The ONLY access is through AKAB MCP tools, which enforce blinding at every step.

## Implementation Methodology

### 1. Fire-and-Forget Scrambling at Startup (Level 3 Only)

When the AKAB Docker container starts:

```python
# 1. Generate scrambled identifiers for all provider/size combinations
anthropic_xs -> model_7a9f2e
anthropic_m  -> model_3b8d1c
openai_l     -> model_9e4f6a
# ... etc for all combinations

# 2. Save mapping in /krill/scrambling/session_[timestamp].json
# This location is COMPLETELY INACCESSIBLE to LLMs

# 3. AKAB Level 3 operates only with scrambled names in all tool responses
```

### 2. Complete Access Control

```
/krill/                      # OUTSIDE allowed_directories
├── scrambling/              # Session-based model mappings
├── campaigns/               # All campaign levels
│   ├── quick/               # Level 1 results
│   ├── standard/            # Level 2 campaigns
│   └── experiments/         # Level 3 campaigns
├── experiments/             # Experiment definitions
└── results/                 # Analysis and raw data

The LLM cannot:
- filesystem:read_file("/krill/...")      ❌ Forbidden
- filesystem:list_directory("/krill/...")  ❌ Forbidden
- docker:exec("cat /krill/...")           ❌ Must be disabled
- ANY direct access method                 ❌ All blocked

The LLM can only:
- akab:quick_compare(...)                 ✓ Level 1 (no blinding)
- akab:create_campaign(...)               ✓ Level 2 (execution blinding)
- akab:create_experiment(...)             ✓ Level 3 (full blinding)
- akab:analyze_results(...)               ✓ Blinded per level
- akab:reveal_experiment(...)             ✓ Only after significance
- akab:diagnose_experiment(...)           ✓ Debug without breaking blind
```

### 3. Experimental Design Phase

Researchers design experiments using only abstract identifiers:

```python
# Level 1 - Quick Compare (see everything)
result = akab:quick_compare(
    prompt="Write a haiku",
    providers=["anthropic", "openai"]
)
# Returns actual provider names and responses

# Level 2 - Campaign (blinded execution)
campaign = akab:create_campaign(
    name="Haiku Quality",
    variants=[{"provider": "anthropic", "model": "claude-3-5-sonnet", ...}],
    success_criteria={
        "primary_metric": "quality_score",
        "direction": "maximize"
    }
)
# Execution shows variant_xxx, can unlock later

# Level 3 - Experiment (full blinding)
available = akab:list_scrambled_models()
# Returns: {
#   "small": ["model_7a9f2e", "model_3b8d1c"],
#   "medium": ["model_9e4f6a", "model_2d5e8b"],
#   "large": ["model_6c3f9d", "model_1a7b4e"]
# }
experiment = akab:create_experiment(
    name="Production Model Selection",
    hypothesis="Larger models produce higher quality technical documentation",
    variants=["model_9e4f6a", "model_6c3f9d"],  # medium vs large
    prompts=[
        "Explain REST API design principles",
        "Write a Python quicksort implementation",
        "Describe microservices architecture"
    ],
    iterations_per_prompt=10,
    success_criteria={
        "primary_metric": "quality_score",
        "evaluation_method": "llm_judge",
        "significance_level": 0.05,
        "effect_size_threshold": 0.3,
        "minimum_iterations": 30
    }
)
# No way to know which is which!
```

### 4. Execution Phase

When tests run:

1. **Level 1**: Direct execution with clear names
2. **Level 2**: AKAB blinds during execution, stores mapping
3. **Level 3**: AKAB maps scrambled IDs to real models internally
4. All results stored in `/krill/` (inaccessible to LLM)

### 5. Analysis Phase

Statistical analysis proceeds with complete objectivity:

```python
# Level 2 - Campaign analysis (blinded)
results = akab:analyze_results(campaign_id)
# Shows variant_xxx results

# Level 3 - Experiment analysis (scrambled)  
results = akab:analyze_results(experiment_id)
# Returns:
{
    "variant_results": {
        "model_9e4f6a": {
            "execution_time": {...},
            "quality_score": {
                "mean": 87.3,
                "trimmed_mean": 88.1,
                "std_dev": 5.2
            },
            "prompt_stats": {...}
        },
        "model_6c3f9d": {
            "execution_time": {...},
            "quality_score": {
                "mean": 91.7,
                "trimmed_mean": 92.3,
                "std_dev": 3.8
            },
            "prompt_stats": {...}
        }
    },
    "winner": null,  # Not revealed yet!
    "experiment_blinded": true,
    "completion_status": {
        "minimum_iterations_met": true,
        "total_tests": 60,
        "ready_for_reveal": true
    }
}
```

### 6. Diagnostic Phase (When Experiments Fail)

When experiments don't reach significance, researchers need insights:

```python
# Safe diagnosis (maintains blinding)
diagnosis = akab:diagnose_experiment(experiment_id)
# Returns:
{
    "statistical_analysis": {
        "significant": false,
        "comparisons": [{
            "t_test": {"p_value": 0.23, "significant": false},
            "effect_size": {"cohens_d": 0.12, "interpretation": "negligible"}
        }]
    },
    "issues_detected": [
        {
            "type": "models_too_similar",
            "severity": "high",
            "description": "Models performing too similarly",
            "recommendation": "Test more diverse model sizes"
        }
    ],
    "response_samples": {
        "model_9e4f6a": [/* anonymized samples */],
        "model_6c3f9d": [/* anonymized samples */]
    }
}

# Force reveal (breaks protocol - use with caution!)
diagnosis = akab:diagnose_experiment(experiment_id, force_reveal=True)
# NOW includes:
{
    "force_reveal_warning": "WARNING: Breaking double-blind protocol",
    "force_revealed_mappings": {
        "model_9e4f6a": {
            "provider": "anthropic",
            "size": "m",
            "model": "claude-3-5-sonnet-20241022"
        },
        "model_6c3f9d": {
            "provider": "openai",
            "size": "l",
            "model": "gpt-4-turbo"
        }
    }
}
```

### 7. Revelation Phase

Only after statistical significance is achieved:

```python
# Level 3 - First reveal experiment results
revealed = akab:reveal_experiment(experiment_id)
# Returns:
{
    "winner": {
        "scrambled_id": "model_6c3f9d",
        "provider": "openai",
        "model": "gpt-4-turbo",
        "size": "l"
    },
    "all_mappings": {
        "model_9e4f6a": {
            "provider": "anthropic",
            "size": "m",
            "model": "claude-3-5-sonnet-20241022"
        },
        "model_6c3f9d": {
            "provider": "openai",
            "size": "l",
            "model": "gpt-4-turbo"
        }
    },
    "statistical_results": {
        "significant": true,
        "best_variant": "model_6c3f9d",
        "comparisons": [...]
    }
}
```

### 8. Archiving Phase (NEW)

Once campaigns or experiments are complete, use the unified `akab_unlock` to archive:

```python
# Level 2 - Unlock and archive campaign
unlocked = akab:akab_unlock(campaign_id)
# Returns:
{
    "id": "campaign_123",
    "name": "Speed Test",
    "type": "campaign",
    "level": 2,
    "mappings": {
        "variant_a": {
            "blinded_id": "variant_xxx",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet"
        }
    },
    "archive": {
        "status": "success",
        "location": "/krill/archive/campaign_123"
    }
}

# Level 3 - Unlock and archive revealed experiment
unlocked = akab:akab_unlock(experiment_id)
# Returns:
{
    "id": "experiment_456",
    "name": "Production Model Selection",
    "type": "experiment",
    "level": 3,
    "hypothesis": "Larger models produce higher quality",
    "winner": "model_6c3f9d",
    "revealed_mappings": {...},
    "archive": {
        "status": "success",
        "location": "/krill/archive/experiment_456"
    }
}

# Archive structure created:
/krill/archive/<id>/
├── blinded/          # State before unlock/reveal
│   ├── campaign.json or experiment.json
│   └── results/
├── clear/            # State after unlock with mappings
│   ├── campaign.json or experiment.json
│   └── results/
└── metadata.json     # Archive metadata
```

#### Unlock Rules

- **Level 1**: No unlocking needed (never blinded)
- **Level 2**: Can unlock anytime after execution
- **Level 3**: Must be revealed first (statistical significance achieved)

If you try to unlock an incomplete or unrevealed experiment:

```python
# Incomplete experiment
error = akab:akab_unlock(incomplete_experiment_id)
# Returns: "Cannot unlock ongoing experiment: 15/30 tests completed"

# Complete but unrevealed experiment  
error = akab:akab_unlock(unrevealed_experiment_id)
# Returns: "Experiment complete but not revealed. Use akab_reveal_experiment first."
```

## Scientific Rigor and Integrity

### Statistical Requirements

Level 3 experiments enforce rigorous statistical standards:

1. **Minimum Sample Size**: Default 30 iterations minimum
2. **Statistical Significance**: p-value < 0.05 (configurable)
3. **Effect Size**: Cohen's d > 0.2 (meaningful difference)
4. **Multiple Comparisons**: Corrected for false discovery
5. **Quality Evaluation**: LLM judge for qualitative assessment

### Quality Evaluation with LLM Judge

When `evaluation_method: "llm_judge"` is specified:

```python
# AKAB uses Claude to evaluate response quality
evaluation_criteria = {
    "relevance": "How well does the response address the prompt?",
    "clarity": "How clear and well-structured is the response?",
    "completeness": "How thorough and complete is the response?",
    "accuracy": "How factually accurate is the response?",
    "helpfulness": "How helpful would this response be to a user?"
}

# Each response scored 0-100 on each criterion
# Overall quality score used for statistical analysis
```

### Common Failure Modes and Diagnostics

The diagnostic tool detects:

1. **Insufficient Data**
   - Not enough iterations run
   - Solution: Run more tests

2. **High Variance**
   - Inconsistent results (CV > 50%)
   - Solution: More iterations or investigate outliers

3. **Models Too Similar**
   - Effect sizes negligible (< 0.2)
   - Solution: Test more diverse models or tasks

4. **Prompt-Specific Issues**
   - Some prompts failing frequently
   - Solution: Review prompt clarity or constraints

## Best Practices

### Choosing the Right Level

1. **Use Level 1 (Quick Compare) for:**
   - Prompt engineering iterations
   - Debugging specific issues
   - Rapid prototyping
   - Personal preference decisions

2. **Use Level 2 (Campaign) for:**
   - A/B testing specific features
   - Comparing known approaches
   - Production deployment decisions
   - Cost/performance optimization

3. **Use Level 3 (Experiment) for:**
   - Model selection for production
   - Research publications
   - High-stakes decisions
   - Eliminating all bias

### Experimental Design Tips

1. **Start Small**: Use Level 1 to refine prompts before Level 3
2. **Power Analysis**: Calculate required sample size beforehand
3. **Diverse Tasks**: Test multiple prompt types for robustness
4. **Clear Hypotheses**: Define what "better" means before starting
5. **Document Everything**: Keep detailed experimental logs

### Handling Failed Experiments

1. **First, diagnose without revealing:**

   ```python
   diagnosis = akab:diagnose_experiment(experiment_id)
   ```

2. **Address identified issues:**
   - Add more iterations if needed
   - Try more diverse models
   - Refine prompts or criteria

3. **Only force reveal if abandoning:**

   ```python
   # Last resort - breaks blinding!
   diagnosis = akab:diagnose_experiment(experiment_id, force_reveal=True)
   ```

## Conclusion

AKAB's three-level architecture provides appropriate rigor for every use case:

1. **Level 1 - Quick Compare**: Fast iteration with full visibility
2. **Level 2 - Campaign**: Proper A/B testing with debug capability
3. **Level 3 - Experiment**: Perfect blinding through complete isolation

By placing all data in `/krill/` outside LLM-accessible paths and providing access only through controlled MCP tools, Level 3 experiments achieve true scientific rigor where:

- **Bias is impossible**: LLMs cannot access model identities
- **Research is natural**: The blinded path is the only path
- **Integrity is maintained**: Results reflect true performance
- **Diagnosis is available**: Understand failures without breaking protocol

The system adapts to your needs - from quick debugging to publication-quality research. Great science happens when the tools make doing the right thing easier than doing the wrong thing. AKAB achieves this through progressive rigor: use only as much blinding as your use case requires.

Remember: The goal isn't to make cheating impossible (humans with Docker access always can), but to make honest, unbiased research the natural default. AKAB succeeds by making the scientific path the easiest path.
