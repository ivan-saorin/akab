"""
Tests for AKAB FileSystem Manager
"""

import pytest
import json
from pathlib import Path
from akab.filesystem import FileSystemManager


@pytest.mark.asyncio
async def test_filesystem_initialization(test_data_dir):
    """Test FileSystemManager initialization"""
    fs = FileSystemManager(str(test_data_dir))
    
    # Check directories were created
    assert fs.base_path.exists()
    assert fs.campaigns_dir.exists()
    assert fs.experiments_dir.exists()
    assert fs.kb_dir.exists()
    assert fs.templates_dir.exists()
    assert fs.results_dir.exists()


@pytest.mark.asyncio
async def test_campaign_save_and_load(test_data_dir, sample_campaign):
    """Test saving and loading campaigns"""
    fs = FileSystemManager(str(test_data_dir))
    
    # Save campaign
    success = await fs.save_campaign(sample_campaign)
    assert success is True
    
    # Check file exists
    campaign_file = fs.campaigns_dir / f"{sample_campaign['id']}.json"
    assert campaign_file.exists()
    
    # Load campaign
    loaded = await fs.load_campaign(sample_campaign['id'])
    assert loaded is not None
    assert loaded['id'] == sample_campaign['id']
    assert loaded['name'] == sample_campaign['name']
    assert loaded['total_experiments'] == sample_campaign['total_experiments']


@pytest.mark.asyncio
async def test_experiment_save_and_load(test_data_dir, sample_campaign):
    """Test saving and loading experiments"""
    fs = FileSystemManager(str(test_data_dir))
    
    # Save campaign first
    await fs.save_campaign(sample_campaign)
    
    # Save experiment
    campaign_id = sample_campaign['id']
    experiment_id = "exp_001"
    config = {
        "experiment_id": experiment_id,
        "campaign_id": campaign_id,
        "provider": "anthropic-local"
    }
    prompt = "Test prompt for experiment"
    result = {
        "response": "Test response",
        "evaluation": {"score": 8.5}
    }
    
    success = await fs.save_experiment(
        campaign_id,
        experiment_id,
        config,
        prompt,
        result
    )
    assert success is True
    
    # Check files exist
    exp_dir = fs.get_experiment_dir(campaign_id, experiment_id)
    assert exp_dir.exists()
    assert (exp_dir / "config.json").exists()
    assert (exp_dir / "prompt.md").exists()
    assert (exp_dir / "result.json").exists()
    
    # Load experiment
    loaded = await fs.load_experiment(campaign_id, experiment_id)
    assert loaded is not None
    assert loaded['id'] == experiment_id
    assert loaded['config']['provider'] == "anthropic-local"
    assert loaded['prompt'] == prompt
    assert loaded['result']['response'] == "Test response"


@pytest.mark.asyncio
async def test_list_campaigns(test_data_dir, sample_campaign):
    """Test listing campaigns"""
    fs = FileSystemManager(str(test_data_dir))
    
    # Initially empty
    campaigns = await fs.list_campaigns()
    assert len(campaigns) == 0
    
    # Save a campaign
    await fs.save_campaign(sample_campaign)
    
    # Should have one campaign
    campaigns = await fs.list_campaigns()
    assert len(campaigns) == 1
    assert campaigns[0]['id'] == sample_campaign['id']
    assert campaigns[0]['name'] == sample_campaign['name']
    
    # Save another campaign
    campaign2 = sample_campaign.copy()
    campaign2['id'] = "test-campaign-2"
    campaign2['name'] = "Test Campaign 2"
    await fs.save_campaign(campaign2)
    
    # Should have two campaigns
    campaigns = await fs.list_campaigns()
    assert len(campaigns) == 2


@pytest.mark.asyncio
async def test_current_campaign_management(test_data_dir):
    """Test current campaign ID management"""
    fs = FileSystemManager(str(test_data_dir))
    
    # Default campaign
    assert fs.get_current_campaign_id() == "default-campaign"
    
    # Set new campaign
    fs.set_current_campaign_id("new-campaign")
    assert fs.get_current_campaign_id() == "new-campaign"


@pytest.mark.asyncio
async def test_meta_prompt_loading(test_data_dir):
    """Test meta prompt loading"""
    fs = FileSystemManager(str(test_data_dir))
    
    # Should return default prompt when file doesn't exist
    prompt = await fs.load_meta_prompt()
    assert "AKAB Experiment Execution Protocol" in prompt
    
    # Create custom meta prompt
    custom_prompt = "# Custom Meta Prompt\n\nCustom instructions here."
    meta_prompt_path = fs.base_path / "meta_prompt.md"
    meta_prompt_path.write_text(custom_prompt)
    
    # Should load custom prompt
    prompt = await fs.load_meta_prompt()
    assert prompt == custom_prompt


@pytest.mark.asyncio
async def test_knowledge_base_loading(test_data_dir):
    """Test knowledge base loading"""
    fs = FileSystemManager(str(test_data_dir))
    
    # Non-existent KB
    kb = await fs.load_knowledge_base("nonexistent")
    assert kb is None
    
    # Create KB file
    kb_content = "# Test Knowledge Base\n\nTest content."
    kb_path = fs.kb_dir / "test_kb.md"
    kb_path.write_text(kb_content)
    
    # Load with .md extension
    kb = await fs.load_knowledge_base("test_kb.md")
    assert kb == kb_content
    
    # Load without extension
    kb = await fs.load_knowledge_base("test_kb")
    assert kb == kb_content


@pytest.mark.asyncio
async def test_results_saving(test_data_dir):
    """Test saving analysis results"""
    fs = FileSystemManager(str(test_data_dir))
    
    campaign_id = "test-campaign"
    analysis = {
        "campaign_name": "Test Campaign",
        "analysis_date": "2025-01-01T00:00:00",
        "total_experiments": 10,
        "summary": "Test analysis summary",
        "provider_metrics": {
            "anthropic-local": {"avg_score": 8.5}
        },
        "recommendations": ["Test recommendation"]
    }
    
    # Save results
    success = await fs.save_results(campaign_id, analysis)
    assert success is True
    
    # Check files exist
    results_dir = fs.results_dir / campaign_id
    assert results_dir.exists()
    assert (results_dir / "analysis.json").exists()
    assert (results_dir / "report.md").exists()
    
    # Verify report content
    report_path = results_dir / "report.md"
    report_content = report_path.read_text()
    assert "Test Campaign" in report_content
    assert "Test analysis summary" in report_content
    assert "anthropic-local" in report_content
    assert "Test recommendation" in report_content
