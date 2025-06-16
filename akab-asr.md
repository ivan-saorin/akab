# AKAB Architecture Significant Records (ASR) - Final

AKAB - Adaptive Knowledge Acquisition Benchmark (was SynapseForge)

## ASR-001: Remote MCP Server Architecture

**Status**: APPROVED  
**Date**: 2025-06-15  
**Author**: Ivan Saorin  
**Reviewers**: Claude Sonnet 4  

### Context

Building systematic AI experimentation platform requiring scalable, remote execution with complete user control and article-worthy documentation.

### Decision

Implement **twin remote MCP servers** (AKAB + Synapse) using FastMCP framework with filesystem-based storage.

### Rationale

#### Technical Drivers

- **Scalability**: Remote deployment allows unlimited concurrent experiments
- **Control**: Filesystem storage provides complete data ownership and portability
- **Separation**: Distinct servers for experimentation vs execution prevents coupling
- **Integration**: MCP protocol enables seamless Claude Desktop integration

#### Business Drivers  

- **Content Creation**: Architecture supports article series documentation
- **Research Velocity**: Systematic AB testing at scale for AI prompt research
- **Knowledge Sharing**: Reproducible experiments with public documentation

### Consequences

#### Positive

- ✅ Complete experiment reproducibility
- ✅ Horizontal scaling capabilities  
- ✅ Platform-agnostic deployment (Windows dev → Ubuntu prod)
- ✅ Rich documentation artifacts for content creation
- ✅ Version control for all experimental data

#### Negative

- ❌ Additional infrastructure complexity vs local tools
- ❌ Network dependency for execution
- ❌ Initial development overhead

#### Risks & Mitigations

- **Risk**: Network latency affecting experiment velocity
  - **Mitigation**: Local development environment mirrors production
- **Risk**: Data loss from filesystem-only storage
  - **Mitigation**: Git-based versioning + regular backups

### Implementation

- **Phase 1**: AKAB MCP server with Docker containerization
- **Phase 2**: Synapse MCP server with program import resolution
- **Phase 3**: CapRover production deployment with monitoring

---

## ASR-002: Filesystem-Based Data Architecture (UPDATED)

**Status**: APPROVED  
**Date**: 2025-06-15 (Updated: 2025-06-16)  
**Author**: Ivan Saorin  
**Update**: Campaign-level folder organization

### Context

Need persistent, version-controllable storage for experiments, prompts, and results with complete user ownership. Update adds better organization for multi-campaign scenarios.

### Decision

Use **direct filesystem storage** with Docker volume mounts and **campaign-level folder hierarchy**.

### Rationale

#### Why Filesystem Over Database

- **Transparency**: Direct file access for debugging and manual inspection
- **Portability**: Easy backup, restore, and migration between environments
- **Version Control**: Git integration for experimental history and collaboration
- **Simplicity**: No ORM complexity, schema migrations, or database administration
- **Content Creation**: Raw files perfect for article screenshots and examples

#### Data Format Strategy

- **Structured Data**: JSON for machine-readable configurations and results
- **Content**: Markdown for prompts (supports code blocks, formatting, metadata)
- **Binary**: Direct file storage for artifacts and media

### File Structure Design (UPDATED)

```
/data/AKAB/
├── campaigns/           # Campaign definitions
├── experiments/         # Campaign-organized experiments
│   ├── {campaign-name}/
│   │   ├── exp_001/
│   │   │   ├── config.json
│   │   │   ├── prompt.md
│   │   │   └── result.json
│   │   └── exp_002/
│   └── {another-campaign}/
├── knowledge_bases/     # Reusable knowledge components  
├── templates/          # Prompt and configuration templates
├── results/            # Campaign-level analysis
│   └── {campaign-name}/
│       ├── analysis.json
│       └── report.md
└── artifacts/          # Generated content and media
```

### Consequences

#### Positive

- ✅ Zero vendor lock-in
- ✅ Human-readable experiment history
- ✅ Perfect for documentation screenshots
- ✅ Simple backup and disaster recovery
- ✅ Git-based collaboration workflows
- ✅ **NEW**: Clear campaign separation and organization

#### Negative  

- ❌ No ACID transactions for concurrent writes
- ❌ Manual consistency management required
- ❌ No built-in query capabilities

#### Mitigations

- File locking for concurrent access safety
- Atomic write operations using temp files
- Structured naming conventions for discoverability
- **NEW**: Campaign-level operations for bulk management

---

## ASR-003: MCP Tool Interface Design (UPDATED)

**Status**: APPROVED  
**Date**: 2025-06-15 (Updated: 2025-06-16)  
**Author**: Ivan Saorin  
**Update**: Remote execution and batch processing tools

### Context

MCP tools must be intuitive for both automated execution and manual article documentation. Update adds support for remote provider execution.

### Decision

Implement **workflow-oriented tool interface** with meta-prompt self-configuration and **dual-mode execution** (local/remote).

### AKAB Tool Specification (UPDATED)

```python
# Meta-configuration
AKAB_get_meta_prompt() → str           # Self-configuring instructions

# Experiment lifecycle
AKAB_get_next_experiment() → dict      # Queue management
AKAB_get_exp_prompt(exp_id) → str      # Prompt retrieval  
AKAB_save_exp_result(exp_id, result, metadata) → dict  # Result storage

# Campaign management
AKAB_get_campaign_status() → dict      # Progress tracking
AKAB_create_campaign(config) → dict    # Campaign initialization
AKAB_analyze_results(campaign_id) → dict  # Aggregated analysis
AKAB_list_campaigns() → dict           # List all campaigns
AKAB_switch_campaign(campaign_id) → dict  # Change active campaign
AKAB_get_current_campaign() → dict     # Current campaign info

# Remote execution (NEW)
AKAB_batch_execute_remote(campaign_id) → dict  # Launch batch execution
AKAB_get_execution_status() → dict     # Monitor remote progress
```

### Design Principles

#### Workflow-First

Tools model the **natural experimentation workflow**:

1. Get next experiment → 2. Execute → 3. Save results → 4. Analyze

#### Dual-Mode Execution (NEW)

- **Local**: Returns prompts for MCP client execution
- **Remote**: Executes via API and auto-saves results

#### Self-Documenting

Each tool response includes:

- Current state context
- Next recommended actions  
- Progress indicators
- Error guidance
- **NEW**: Cost estimates for remote execution

#### Article-Friendly

Tool interactions designed for:

- Clear screenshot opportunities
- Step-by-step narrative flow
- Before/after comparisons
- Progress visualization
- **NEW**: Cost/performance comparisons

### Consequences

#### Positive

- ✅ Intuitive learning curve for new users
- ✅ Natural documentation flow for articles
- ✅ Consistent interaction patterns
- ✅ Built-in progress tracking
- ✅ **NEW**: Seamless local/remote execution
- ✅ **NEW**: True multi-model AB testing capability

#### Negative

- ❌ More tools than minimal interface
- ❌ Workflow assumptions may not fit all use cases
- ❌ **NEW**: Complexity of dual-mode handling

---

## ASR-008: Multi-Provider Remote Execution Architecture

**Status**: APPROVED  
**Date**: 2025-06-16  
**Author**: Ivan Saorin  

### Context

Need to support multiple LLM providers (OpenAI, Anthropic, Google, etc.) for comprehensive AB testing and cost optimization.

### Decision

Implement **provider abstraction layer** with Instructor library for structured outputs and unified experiment interface.

### Architecture Design

#### Provider Types

```python
class ProviderType(Enum):
    LOCAL = "local"      # MCP client execution
    REMOTE = "remote"    # API-based execution
```

#### Provider Framework

- Base abstract class for all providers
- Instructor integration for structured outputs
- Unified result schema across providers
- Cost tracking and estimation

#### Supported Providers

1. **Local** (anthropic-local): Claude Desktop/Cline execution
2. **Anthropic API**: Claude models via API
3. **OpenAI**: GPT-3.5/4 models
4. **Google**: Gemini Pro models
5. **Mistral**: Open weight models
6. **Future**: Easy extension for new providers

### Rationale

#### Why Multiple Providers

- **Comparison**: True AB testing across models
- **Cost Optimization**: Route by task requirements
- **Resilience**: Fallback options
- **Feature Access**: Provider-specific capabilities
- **Research**: Scientific model comparison

#### Why Instructor

- **Structured Output**: Consistent results across providers
- **Type Safety**: Pydantic validation
- **Error Handling**: Unified error responses
- **Documentation**: Self-documenting schemas

### Implementation Strategy

#### Phase 1: Core Framework

- Provider base class
- Local provider (existing flow)
- Provider manager/registry

#### Phase 2: Remote Providers

- OpenAI with Instructor
- Anthropic API provider
- Cost calculation logic

#### Phase 3: Advanced Features

- Provider health checks
- Automatic fallback
- Parallel provider execution

### Consequences

#### Positive

- ✅ True multi-model experimentation platform
- ✅ Cost transparency and optimization
- ✅ Scientific comparison capabilities
- ✅ Future-proof provider additions
- ✅ Structured, validated outputs

#### Negative

- ❌ Additional API dependencies
- ❌ Provider-specific quirks to handle
- ❌ Cost management complexity
- ❌ Rate limit coordination

#### Mitigations

- Comprehensive error handling
- Provider abstraction hides complexity
- Built-in rate limiting
- Cost warnings and limits

---

## ASR-009: Batch Remote Execution Design

**Status**: APPROVED  
**Date**: 2025-06-16  
**Author**: Ivan Saorin  

### Context

Remote API experiments need efficient batch execution for AB testing at scale while maintaining cost control.

### Decision

Implement **single active remote campaign** constraint with background batch execution and monitoring tools.

### Design Principles

#### Single Campaign Execution

- Only one remote campaign active at a time
- Prevents resource conflicts and cost surprises
- Clear execution ownership
- Simple mental model

#### Background Processing

- Non-blocking campaign launch
- Persistent progress tracking
- Docker container lifecycle management
- Graceful shutdown capability

#### Monitoring and Control

```python
# Launch execution
AKAB_batch_execute_remote(campaign_id)

# Monitor progress
AKAB_get_execution_status()
# Returns: progress, ETA, costs

# Stop execution
# Via Docker container stop
```

### Implementation Details

#### State Management

- Global active campaign tracker
- Progress persistence across restarts
- Atomic state updates

#### Execution Loop

1. Launch background task
2. Sequential experiment execution
3. Automatic result storage
4. Progress reporting
5. Cost accumulation

#### Error Handling

- Continue on experiment failure
- Log errors for analysis
- Report final success/failure count

### Consequences

#### Positive

- ✅ Predictable execution model
- ✅ Cost control and visibility
- ✅ Simple operational model
- ✅ Clear progress tracking
- ✅ Efficient batch processing

#### Negative

- ❌ No parallel campaign execution
- ❌ Docker stop as only halt mechanism
- ❌ Sequential processing limitations

#### Mitigations

- Clear status reporting
- ETA calculations
- Optional parallel execution within campaign
- Future: campaign queueing system

---

## ASR-010: Cost Management and Safety

**Status**: APPROVED  
**Date**: 2025-06-16  
**Author**: Ivan Saorin  

### Context

Remote API execution can incur significant costs. Users need transparency and control over expenses.

### Decision

Implement **proactive cost estimation** with warnings at 20+ experiments and detailed tracking.

### Cost Control Features

#### Estimation

- Pre-execution cost calculation
- Per-provider pricing models
- Token usage estimates
- Conservative estimates (round up)

#### Warnings

```python
if num_experiments > 20:
    warning = f"⚠️ About to execute {num} experiments. 
               Estimated cost: ${estimate:.2f}"
```

#### Tracking

- Per-experiment cost recording
- Campaign total accumulation
- Provider comparison metrics
- Cost/quality ratio analysis

### Implementation

#### Provider Costs

```python
COST_PER_1K_TOKENS = {
    "gpt-4-turbo": 0.03,
    "gpt-3.5-turbo": 0.002,
    "claude-3-opus": 0.025,
    "claude-3-sonnet": 0.003,
    "gemini-pro": 0.001
}
```

#### Safeguards

- No hard limits (research flexibility)
- Clear warnings before execution
- Real-time cost tracking
- Post-execution reports

### Consequences

#### Positive

- ✅ No surprise bills
- ✅ Informed decision making
- ✅ Cost-aware research
- ✅ Budget planning capability

#### Negative

- ❌ Estimation complexity
- ❌ Provider pricing changes
- ❌ No hard stop mechanism

---

## Implementation Roadmap (UPDATED)

### Phase 1: Foundation (COMPLETE)

- [x] Base MCP server implementation with FastMCP
- [x] Docker containerization with volume mounts
- [x] Basic filesystem operations and API key auth
- [x] Local development environment setup
- [x] Campaign safety protocol

### Phase 2: Core Features (COMPLETE)  

- [x] Complete AKAB tool implementation
- [x] Campaign and experiment management
- [x] Meta-prompt self-configuration system
- [x] Comprehensive error handling and logging

### Phase 3: Remote Providers (CURRENT)

- [ ] Provider base class and abstraction
- [ ] Local provider adapter
- [ ] OpenAI provider with Instructor
- [ ] Batch execution implementation
- [ ] Campaign folder reorganization
- [ ] Cost tracking and warnings

### Phase 4: Production Enhancement

- [ ] Additional providers (Anthropic API, Google)
- [ ] CapRover deployment configuration
- [ ] Performance optimization and caching
- [ ] Advanced analysis tools

### Phase 5: Content Creation

- [ ] Article series with remote execution examples
- [ ] AB testing case studies
- [ ] Cost/performance analysis posts
- [ ] Community examples and tutorials

---

## Success Metrics (UPDATED)

### Technical Metrics

- **Uptime**: >99% availability in production
- **Performance**: <500ms overhead for provider selection
- **Reliability**: Zero data loss through filesystem operations
- **Scalability**: Support for 100+ concurrent experiments
- **NEW**: <2s per remote experiment execution overhead

### Content Metrics

- **Article Series**: 7 comprehensive technical articles (expanded)
- **Community Engagement**: GitHub stars, Docker pulls
- **User Adoption**: External users successfully deploying
- **Knowledge Transfer**: Clear reproduction by others
- **NEW**: Published AB testing case studies

### Research Metrics  

- **Experiment Velocity**: 10x improvement over manual testing
- **Data Quality**: Consistent, analyzable experimental results
- **Innovation Discovery**: Documented breakthrough insights
- **Methodology**: Replicable systematic testing approach
- **NEW**: Multi-model comparison studies published

### Business Metrics (NEW)

- **Cost Efficiency**: 50% reduction via provider optimization
- **Provider Coverage**: 5+ major LLM providers supported
- **AB Test Throughput**: 1000+ experiments/day capability
- **Research Impact**: Citations in AI research papers

---

## Architecture Evolution

### From Local Tool to Research Platform

1. **v1.0**: Local MCP experiment runner
2. **v2.0**: Multi-provider AB testing platform
3. **Future**: Industry standard for LLM comparison

### Key Innovations

- **Dual-mode execution**: Seamless local/remote
- **Provider abstraction**: True model agnosticism
- **Structured outputs**: Scientific rigor via Instructor
- **Cost consciousness**: Research with budget awareness

This ASR documentation provides the foundation for AKAB as **the definitive platform for systematic AI research and AB testing**. Every architectural decision supports both immediate research goals and long-term community impact.
