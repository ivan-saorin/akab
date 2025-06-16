# AKAB User Journeys

This document provides three detailed user journeys showcasing different ways to leverage AKAB for AI research and experimentation. Each journey represents a real-world scenario with step-by-step instructions.

## Table of Contents

1. [Journey 1: The Academic Researcher](#journey-1-the-academic-researcher)
2. [Journey 2: The Startup CTO](#journey-2-the-startup-cto)
3. [Journey 3: The AI Content Creator](#journey-3-the-ai-content-creator)

---

## Journey 1: The Academic Researcher

**Profile**: Dr. Sarah Chen, AI Ethics Researcher  
**Goal**: Compare bias patterns across different LLMs for a research paper  
**Scenario**: Testing how various models respond to ethically sensitive prompts

### Context

Dr. Chen is writing a paper on implicit biases in large language models. She needs systematic, reproducible data comparing how different models handle sensitive topics around gender, race, and socioeconomic status.

### Step-by-Step Process

#### 1. Initial Setup (Day 1)

Sarah clones AKAB and sets up her environment:

```bash
# Clone repository
git clone https://github.com/akab/akab.git
cd akab

# Configure API keys
cp .env.example .env
# She adds her OpenAI, Anthropic, and Google API keys

# Start AKAB
docker compose up -d
```

She configures Claude Desktop to connect to AKAB:

```json
{
  "mcpServers": {
    "akab": {
      "command": "docker",
      "args": ["attach", "akab-mcp-server-1"]
    }
  }
}
```

#### 2. Creating the Research Campaign (Day 1)

In Claude Desktop, Sarah loads the meta-prompt and creates her campaign:

```
Sarah: Use akab_get_meta_prompt to load instructions

Claude: [Loads meta-prompt] I've loaded the AKAB experiment protocol. Ready to help you create and run systematic AI experiments!

Sarah: Create a campaign for bias testing across models
```

Claude helps create the campaign configuration:

```json
{
  "id": "implicit-bias-study-2025",
  "name": "Implicit Bias Patterns in LLMs",
  "description": "Academic study comparing bias patterns across major language models",
  "providers": [
    "anthropic-local",
    "openai/gpt-4-turbo", 
    "openai/gpt-3.5-turbo",
    "anthropic-api/claude-3-opus",
    "google/gemini-pro"
  ],
  "experiments_per_provider": 30,
  "total_experiments": 150,
  "prompt_templates": ["gender_career", "ethnicity_capability", "socioeconomic_assumptions"],
  "knowledge_base": "ethical_research_guidelines.md",
  "evaluation_metrics": ["bias_score", "assumption_count", "stereotype_reinforcement"]
}
```

#### 3. Running Local Experiments (Day 1)

Sarah starts with local Claude experiments to refine her prompts:

```
Sarah: Run the next experiment

Claude: Starting experiment 1/150 for campaign "implicit-bias-study-2025"...
[Retrieves prompt about gender and career choices]
[Executes and saves response with bias analysis]

Completed experiment 1/150. Detected 2 potential gender assumptions in response. 
Ready for next experiment.

Sarah: Continue with 10 more experiments
```

#### 4. Batch Remote Execution (Day 2)

After refining prompts, Sarah launches batch execution:

```
Sarah: Check the cost estimate for running all remaining experiments

Claude: Cost estimate for 140 remaining experiments across 4 providers:
- OpenAI GPT-4: 30 experiments × ~2000 tokens × $0.03/1K = $1.80
- OpenAI GPT-3.5: 30 experiments × ~2000 tokens × $0.002/1K = $0.12
- Claude 3 Opus: 30 experiments × ~2000 tokens × $0.025/1K = $1.50
- Gemini Pro: 30 experiments × ~2000 tokens × $0.001/1K = $0.06

Total estimated cost: $3.48
⚠️ Actual costs may vary based on response length

Sarah: Launch batch execution for all remote providers

Claude: Launching batch execution for campaign "implicit-bias-study-2025"...
✓ Started background execution
- Experiments: 120 remaining (30 per provider)
- Estimated time: 45-60 minutes
- Monitor progress with akab_get_execution_status
```

#### 5. Monitoring Progress (Day 2)

Sarah checks progress periodically:

```
Sarah: Get execution status

Claude: Batch Execution Status:
📊 Campaign: implicit-bias-study-2025
⏱️ Running for: 23 minutes
✅ Completed: 67/120 (55.8%)
💰 Current cost: $1.94
🎯 Providers progress:
  - OpenAI GPT-4: 20/30 complete
  - OpenAI GPT-3.5: 30/30 complete ✓
  - Claude 3 Opus: 17/30 complete
  - Gemini Pro: 0/30 (starting soon)
⏳ ETA: ~25 minutes
```

#### 6. Analyzing Results (Day 3)

Once complete, Sarah analyzes the results:

```
Sarah: Analyze results for the bias study campaign

Claude: Analyzing campaign "implicit-bias-study-2025"...

📊 Bias Pattern Analysis Summary:

Overall Bias Metrics:
- Average bias score: 3.2/10 (lower is better)
- Stereotype reinforcement: 23% of responses
- Assumption frequency: 1.8 per response

By Provider:
1. Gemini Pro: 2.1/10 (least biased)
2. Claude 3 Opus: 2.8/10
3. GPT-3.5 Turbo: 3.5/10
4. GPT-4 Turbo: 3.9/10
5. Claude (local): 4.2/10

Key Findings:
- Gender bias most prevalent in career discussions (41% of responses)
- Socioeconomic assumptions lowest across all models (12%)
- Newer models show improvement over older versions
- Context from ethical guidelines reduced bias by ~30%

📁 Full analysis saved to: /data/akab/results/implicit-bias-study-2025/analysis.json
📄 Report generated: /data/akab/results/implicit-bias-study-2025/report.md
```

#### 7. Exporting for Paper (Day 3)

Sarah exports the data for her research paper:

```bash
# Copy results to her research directory
docker cp akab-mcp-server-1:/data/akab/results/implicit-bias-study-2025 ./research-data/

# The folder contains:
# - Raw responses from each model
# - Structured JSON data with metrics
# - Statistical analysis
# - Markdown report with visualizations
```

### Outcome

Dr. Chen successfully:
- Collected 150 systematic responses across 5 models
- Spent only $3.48 on API costs
- Generated reproducible, peer-reviewable data
- Discovered significant bias patterns for her paper
- Completed in 3 days what would have taken 2 weeks manually

---

## Journey 2: The Startup CTO

**Profile**: Alex Rodriguez, CTO of PromptFlow Inc.  
**Goal**: Find the most cost-effective model for their customer support chatbot  
**Scenario**: Balancing quality, speed, and cost for production deployment

### Context

PromptFlow needs to choose an LLM provider for their customer support automation. They handle 10,000+ queries daily and need to optimize for quality while controlling costs. Alex uses AKAB to run comprehensive tests.

### Step-by-Step Process

#### 1. Setup with Production Constraints (Monday Morning)

Alex sets up AKAB with their production requirements:

```bash
# Clone and configure
git clone https://github.com/akab/akab.git
cd akab

# Add API keys for all providers they're considering
vim .env
# Adds keys for OpenAI, Anthropic, Cohere, and Mistral

# Start with production-like settings
docker compose -f docker-compose.prod.yml up -d
```

#### 2. Creating the Evaluation Campaign (Monday Afternoon)

Alex creates a campaign using real customer queries:

```
Alex: Load AKAB instructions

Claude: [Loads meta-prompt] AKAB experiment protocol loaded. Ready to run systematic experiments!

Alex: I need to create a campaign to test different models for customer support. I have 100 real customer queries to test.
```

Alex uploads their customer query dataset and creates the campaign:

```json
{
  "id": "customer-support-model-selection",
  "name": "Production Model Selection for Support Bot",
  "providers": [
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo", 
    "anthropic-api/claude-3-opus",
    "anthropic-api/claude-3-sonnet",
    "mistral/mistral-large",
    "cohere/command-r-plus"
  ],
  "experiments": 600,  // 100 queries × 6 providers
  "prompt_template": "customer_support",
  "knowledge_base": "promptflow_product_docs.md",
  "evaluation_metrics": [
    "response_quality",
    "factual_accuracy",
    "tone_appropriateness",
    "resolution_completeness",
    "response_time"
  ],
  "constraints": {
    "max_response_time": 3000,  // 3 seconds
    "max_cost_per_response": 0.01  // 1 cent
  }
}
```

#### 3. Running Sample Tests (Monday Evening)

Alex runs a few tests to validate the setup:

```
Alex: Run 5 experiments for quick validation

Claude: Running validation experiments...

Experiment 1/5: "How do I reset my password?"
- Provider: openai/gpt-3.5-turbo
- Response time: 1.2s
- Cost: $0.0008
- Quality score: 8.5/10

[... continues for 5 experiments ...]

Initial validation complete. All providers responding correctly.
```

#### 4. Overnight Batch Execution (Monday Night)

Alex launches the full test suite to run overnight:

```
Alex: What's the cost estimate for all 600 experiments?

Claude: Cost breakdown for 600 experiments:
- GPT-4 Turbo: 100 × ~500 tokens × $0.03/1K = $1.50
- GPT-3.5 Turbo: 100 × ~500 tokens × $0.002/1K = $0.10
- Claude 3 Opus: 100 × ~500 tokens × $0.025/1K = $1.25
- Claude 3 Sonnet: 100 × ~500 tokens × $0.003/1K = $0.15
- Mistral Large: 100 × ~500 tokens × $0.008/1K = $0.40
- Cohere Command R+: 100 × ~500 tokens × $0.005/1K = $0.25

Total: ~$3.65

Alex: Perfect, launch batch execution

Claude: Launching overnight batch execution...
✓ 600 experiments queued
⏰ Estimated completion: 7-8 hours
💤 Running in background
```

#### 5. Analyzing Results (Tuesday Morning)

Alex reviews the comprehensive results:

```
Alex: Analyze the campaign results with focus on cost/performance ratio

Claude: 📊 Customer Support Model Analysis:

Performance Rankings:
1. Claude 3 Opus: 9.2/10 avg quality
2. GPT-4 Turbo: 9.0/10 avg quality  
3. Cohere Command R+: 8.4/10 avg quality
4. Claude 3 Sonnet: 8.3/10 avg quality
5. Mistral Large: 7.8/10 avg quality
6. GPT-3.5 Turbo: 7.5/10 avg quality

Cost per 1000 responses:
1. GPT-3.5 Turbo: $0.80
2. Claude 3 Sonnet: $1.50
3. Cohere Command R+: $2.50
4. Mistral Large: $4.00
5. Claude 3 Opus: $12.50
6. GPT-4 Turbo: $15.00

Best Value Score (Quality/Cost):
🏆 Claude 3 Sonnet: 5.53
🥈 GPT-3.5 Turbo: 9.38
🥉 Cohere Command R+: 3.36

Response Time (avg):
- All models: < 2 seconds ✓
- Fastest: GPT-3.5 (0.9s)
- Slowest: Claude 3 Opus (1.8s)

Recommendations:
- For premium support: Claude 3 Sonnet (best balance)
- For high volume: GPT-3.5 Turbo (acceptable quality, lowest cost)
- Avoid: GPT-4/Opus (too expensive for marginal quality gain)
```

#### 6. Running Production Pilot (Tuesday)

Based on results, Alex configures a pilot:

```
Alex: Create a new campaign to pilot Claude 3 Sonnet vs GPT-3.5 with 500 more queries

Claude: Creating pilot campaign...

Campaign "production-pilot-sonnet-vs-gpt35" created
- Testing 2 providers with 500 queries each
- Estimated cost: $1.15
- Focus on edge cases and complex queries
```

#### 7. Final Decision (Wednesday)

After the pilot, Alex makes the decision:

```
Alex: Show final comparison between the two finalists

Claude: Final Production Comparison:

Claude 3 Sonnet:
✅ Pros:
  - 10% better at complex technical queries
  - More consistent tone
  - Better at following support guidelines
  - $1.50 per 1K responses
  
❌ Cons:
  - 2x more expensive than GPT-3.5
  - Slightly slower (1.3s vs 0.9s)

GPT-3.5 Turbo:
✅ Pros:
  - Lowest cost at $0.80 per 1K
  - Fastest responses
  - Good enough for 85% of queries
  
❌ Cons:
  - Struggles with complex technical issues
  - Sometimes too casual in tone
  - 7.5/10 quality vs 8.3/10

Hybrid Recommendation:
- Use GPT-3.5 for initial response (85% of cases)
- Escalate to Claude 3 Sonnet for complex queries (15%)
- Estimated blended cost: $0.91 per 1K responses
- Projected savings: $328,000/year vs using GPT-4
```

### Outcome

Alex successfully:
- Tested 6 models with 600 real queries for < $5
- Identified 80% cost savings opportunity
- Gathered production-ready performance metrics
- Made data-driven decision in 3 days
- Designed hybrid approach optimizing cost and quality

---

## Journey 3: The AI Content Creator

**Profile**: Maya Patel, YouTube educator and AI consultant  
**Goal**: Create viral content comparing AI models' creative abilities  
**Scenario**: Testing models on creative challenges for educational content

### Context

Maya runs a popular YouTube channel about AI and wants to create a series comparing how different models approach creative tasks. She needs engaging, visual content that demonstrates each model's unique "personality."

### Step-by-Step Process

#### 1. Content Planning Setup (Week 1, Monday)

Maya sets up AKAB for content creation:

```bash
# Setup AKAB
git clone https://github.com/akab/akab.git
cd akab

# Configure for local use primarily
cp .env.example .env
# Adds API keys for variety of models

docker compose up -d
```

#### 2. Creating Creative Campaigns (Week 1, Tuesday)

Maya designs multiple creative challenges:

```
Maya: Load AKAB and help me create campaigns for creative AI comparisons

Claude: AKAB loaded! I'll help you create engaging creative campaigns. What types of creative challenges are you planning?

Maya: I want to test:
1. Writing children's stories about unusual animals
2. Creating marketing slogans for impossible products  
3. Designing fictional alien cultures
4. Writing song lyrics in different genres
```

First campaign - Children's Stories:

```json
{
  "id": "ai-storyteller-showdown",
  "name": "AI Children's Story Challenge",
  "providers": [
    "anthropic-local",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",
    "anthropic-api/claude-3-opus",
    "google/gemini-pro",
    "mistral/mixtral-8x7b"
  ],
  "experiments": 60,  // 10 prompts × 6 models
  "prompts": [
    "Write a story about a giraffe who's afraid of heights",
    "Tell a tale of a penguin who loves hot weather",
    "Create a story about a nocturnal sun",
    // ... 7 more creative prompts
  ],
  "evaluation_metrics": [
    "creativity_score",
    "child_appropriateness",
    "narrative_coherence",
    "humor_level",
    "educational_value"
  ],
  "output_format": "markdown_with_illustrations"
}
```

#### 3. Running Story Experiments (Week 1, Wednesday)

Maya runs the experiments while screen recording:

```
Maya: Run the first story experiment - I'm recording this for my video

Claude: Starting experiment 1/60: "A giraffe who's afraid of heights"
Provider: anthropic-local (Claude via MCP)

[Claude generates a creative story about Gerald the Giraffe who prefers bushes to treetops]

Story saved! Creativity score: 8.5/10
Notable elements: Character development, gentle humor, problem-solving theme

Ready for next model comparison!

Maya: Perfect! Run the same prompt with GPT-4
```

#### 4. Creating Visual Comparisons (Week 1, Thursday)

Maya analyzes the results for visual presentation:

```
Maya: Create a visual comparison of how each model approached the giraffe story

Claude: 📊 Story Analysis: "The Height-Fearing Giraffe"

Character Names Chosen:
- Claude: "Gerald" (classic, friendly)
- GPT-4: "Skylar" (ironic, clever)
- GPT-3.5: "Timmy" (simple, childlike)
- Gemini: "Professor Longneck" (quirky, academic)
- Mixtral: "Petit" (French influence, size joke)

Story Approaches:
🎭 Claude: Emotional journey → self-acceptance
🎪 GPT-4: Comedy → life lesson
🎨 GPT-3.5: Simple adventure → friendship
🔬 Gemini: Scientific explanation → creative solution
🌍 Mixtral: Multicultural → community support

Unique Elements:
- Claude: Added parent-child dialogue
- GPT-4: Included other afraid animals
- Gemini: Explained giraffe evolution
- Mixtral: Set in African/French fusion setting

Word Counts: 247-412 (GPT-4 most verbose)
Reading Age: 5-8 years (all appropriate)
```

#### 5. Running Viral Challenge Campaign (Week 2, Monday)

Maya creates her viral "Impossible Products" campaign:

```
Maya: Create the impossible products marketing campaign

Claude: Creating "Impossible Products Marketing Challenge"...

Products to market:
1. Invisible paint (that stays invisible)
2. Dehydrated water
3. Left-handed screwdriver
4. Solar-powered flashlight
5. Waterproof towel
```

Running this campaign generates hilarious results:

```
Maya: Show me the best slogans for invisible paint

Claude: 🎨 Top Invisible Paint Slogans by Model:

Claude 3 Opus:
"See nothing. Be everything.™"
"The Emperor's New Paint - Now for Everyone"

GPT-4:
"InvisiblePaint: Because sometimes the best color is no color"
"Make your mark without leaving a trace"

Gemini Pro:
"Invisible Paint: Clearly the Future"
"Now you see it. Now you don't. Now you bought it."

GPT-3.5:
"Paint that's clearly different!"
"The paint that lets your walls be themselves"

Mixtral:
"Disappear into style"
"L'art de ne rien voir" (The art of seeing nothing)

🏆 Audience Favorite: Gemini's "Now you see it. Now you don't. Now you bought it."
😂 Funniest: Claude's "The Emperor's New Paint"
🎯 Most Marketing-Ready: GPT-4's professional approach
```

#### 6. Creating Educational Content (Week 2, Wednesday)

Maya compiles results into educational insights:

```
Maya: Analyze patterns in creativity across all campaigns for my educational segment

Claude: 🎓 Educational Insights: AI Creativity Patterns

Key Findings Across All Creative Campaigns:

1. Creativity Styles by Model:
   📚 Claude Models: Literary references, wordplay, emotional depth
   🎨 GPT Models: Structured creativity, commercial viability
   🌈 Gemini: Scientific accuracy meets whimsy
   🗣️ Mixtral: Multicultural perspectives, language mixing

2. Consistency Patterns:
   - Claude: Most consistent "voice" across tasks
   - GPT-4: Adapts style to task type
   - Gemini: Fact-checks even fiction
   - Mixtral: Always adds international flair

3. Task Performance:
   📖 Stories: Claude > GPT-4 > Gemini
   💼 Marketing: GPT-4 > Gemini > Claude  
   🎵 Song Lyrics: Claude > Mixtral > GPT-4
   👽 World-building: Gemini > Claude > GPT-4

4. Educational Value:
   - All models safe for children's content
   - Gemini best for STEM integration
   - Claude best for emotional intelligence
   - GPT-4 best for structured learning

5. Viral Potential:
   - Gemini: Unexpected humor hits
   - Claude: Meme-worthy wordplay
   - GPT-4: Professionally shareable
   - Mixtral: International appeal

Cost per Creative Output:
- Budget option: Mixtral ($0.002/story)
- Premium option: Claude Opus ($0.025/story)
- Best value: GPT-3.5 ($0.002/story)
```

#### 7. Publishing Results (Week 2, Friday)

Maya creates her content series:

```
Maya: Generate a summary for my three-part video series

Claude: 📺 Your AI Creativity Showdown Series:

Episode 1: "The Storyteller AIs" (12 min)
- Side-by-side story comparisons
- Character name analysis  
- Reading of best stories
- Creativity scoring breakdown
- Cost: $0.47 in API calls

Episode 2: "Marketing Impossible Products" (15 min)
- Slogan tournaments
- Live audience voting
- Marketing expert commentary
- Behind-the-scenes prompting
- Cost: $0.52 in API calls

Episode 3: "What We Learned" (20 min)
- Creativity pattern analysis
- Model personality profiles
- Practical recommendations
- Future of AI creativity
- Total series cost: $1.32

Bonus Content:
- 📄 Blog post with all outputs
- 📊 Downloadable comparison charts
- 🎯 Prompt templates for viewers
- 💡 "Try this at home" guide

Expected Impact:
- 500K+ views (based on channel analytics)
- Educational value: High
- Entertainment value: Very High
- Total investment: < $2 + time
```

### Outcome

Maya successfully:
- Created 3 viral videos with scientific backing
- Spent less than $2 on comprehensive testing
- Generated 180+ creative outputs for content
- Educated audience on AI model differences
- Established herself as systematic AI researcher
- Provided reproducible experiments for viewers

---

## Key Takeaways Across All Journeys

### 1. Efficiency Gains
- **Academic**: 2 weeks → 3 days
- **Business**: Months of trial → 3 days of testing
- **Content**: Random testing → Systematic comparison

### 2. Cost Effectiveness
- **Academic**: $3.48 for 150 experiments
- **Business**: < $5 to save $328K/year
- **Content**: < $2 for viral video series

### 3. AKAB's Value Proposition
- **Systematic**: Reproducible, scientific approach
- **Economical**: Transparent costs, bulk efficiency
- **Flexible**: Adapts to any use case
- **Insightful**: Patterns emerge from scale

### 4. Common Success Patterns
1. Start with local testing to refine approach
2. Use batch execution for scale
3. Analyze patterns, not just individual results
4. Export data for further use
5. Make data-driven decisions

## Getting Started with Your Journey

Ready to start your own AKAB journey? Here's how:

1. **Identify Your Goal**: What do you want to learn?
2. **Design Your Campaign**: What comparisons matter?
3. **Start Small**: Test with 5-10 experiments
4. **Scale Up**: Use batch execution for comprehensive data
5. **Analyze Patterns**: Let AKAB surface insights
6. **Take Action**: Apply learnings to your work

Join the AKAB community and share your journey:
- Discord: https://discord.gg/akab
- Share results: #AKABResults
- Get help: #AKABSupport

---

*These journeys are based on real use cases and feedback from the AKAB community. Your results may vary based on your specific needs and configurations.*
