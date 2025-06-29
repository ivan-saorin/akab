"""Tests for AKAB MCP server."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from akab.server import AKABServer
from akab.providers import ProviderManager
from substrate import ValidationError


@pytest.fixture
async def akab_server():
    """Create AKAB server instance for testing."""
    server = AKABServer()
    yield server
    # Cleanup
    await server.sampling_manager.close()


class TestAKABServer:
    """Test AKAB server functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, akab_server):
        """Test server initialization."""
        assert akab_server.name == "akab"
        assert akab_server.version == "2.0.0"
        assert akab_server.storage is not None
        assert akab_server.provider_manager is not None
        assert akab_server.comparison_engine is not None
        assert akab_server.campaign_manager is not None
        
    @pytest.mark.asyncio
    async def test_get_capabilities(self, akab_server):
        """Test capabilities reporting."""
        caps = await akab_server.get_capabilities()
        
        assert "features" in caps
        assert len(caps["features"]) > 0
        assert "providers" in caps
        assert "version" in caps
        
    @pytest.mark.asyncio
    async def test_quick_compare_invalid_provider(self, akab_server):
        """Test quick compare with invalid provider."""
        # Mock the tool context
        ctx = Mock()
        
        with pytest.raises(ValidationError) as exc_info:
            await akab_server.mcp._tools["akab_quick_compare"](
                ctx,
                prompt="Test prompt",
                providers=["invalid_provider"],
                parameters=None,
                constraints=None
            )
            
        assert "Unknown provider" in str(exc_info.value)
        
    @pytest.mark.asyncio
    async def test_create_campaign(self, akab_server):
        """Test campaign creation."""
        ctx = Mock()
        
        # Mock provider validation
        akab_server.provider_manager.is_valid_provider = Mock(return_value=True)
        
        result = await akab_server.mcp._tools["akab_create_campaign"](
            ctx,
            name="Test Campaign",
            description="Test description",
            prompts=[{"prompt": "Test prompt"}],
            providers=["anthropic_m", "openai_m"],
            iterations=3,
            constraints=None
        )
        
        assert result["success"] is True
        assert "campaign" in result["data"]
        assert result["data"]["campaign"]["name"] == "Test Campaign"


class TestProviderManager:
    """Test provider manager functionality."""
    
    def test_list_providers(self):
        """Test listing available providers."""
        manager = ProviderManager()
        providers = manager.list_providers()
        
        assert len(providers) > 0
        assert "anthropic_m" in providers
        assert "openai_m" in providers
        
    def test_is_valid_provider(self):
        """Test provider validation."""
        manager = ProviderManager()
        
        assert manager.is_valid_provider("anthropic_m") is True
        assert manager.is_valid_provider("invalid_provider") is False
        
    def test_get_provider_config(self):
        """Test getting provider configuration."""
        manager = ProviderManager()
        
        config = manager.get_provider_config("anthropic_m")
        assert config.name == "anthropic_m"
        assert config.size == "m"
        assert config.model is not None
        assert config.cost_per_1k_input > 0
        
    def test_estimate_cost(self):
        """Test cost estimation."""
        manager = ProviderManager()
        
        cost = manager.estimate_cost("anthropic_m", 1000, 500)
        assert cost > 0
        assert isinstance(cost, float)
        
    def test_get_provider_info(self):
        """Test getting provider information."""
        manager = ProviderManager()
        
        info = manager.get_provider_info("openai_m")
        assert info["name"] == "openai_m"
        assert "model" in info
        assert "max_tokens" in info
        assert "cost_per_1k_input" in info
        assert "api_key_configured" in info


class TestComparisonEngine:
    """Test comparison engine functionality."""
    
    @pytest.mark.asyncio
    async def test_prepare_prompt(self):
        """Test prompt preparation with parameters."""
        from akab.comparison import ComparisonEngine
        
        manager = Mock()
        engine = ComparisonEngine(manager)
        
        prompt = "Hello {{name}}, explain {{topic}}"
        params = {"name": "Alice", "topic": "quantum computing"}
        
        result = engine._prepare_prompt(prompt, params)
        assert result == "Hello Alice, explain quantum computing"
        
    @pytest.mark.asyncio
    async def test_analyze_results(self):
        """Test result analysis."""
        from akab.comparison import ComparisonEngine
        from substrate import ComparisonResult
        
        manager = Mock()
        engine = ComparisonEngine(manager)
        
        results = [
            ComparisonResult(
                provider="provider1",
                response="Response 1",
                latency_ms=100,
                tokens_used=50,
                cost_estimate=0.01
            ),
            ComparisonResult(
                provider="provider2",
                response="Response 2",
                latency_ms=200,
                tokens_used=60,
                cost_estimate=0.02
            ),
        ]
        
        analysis = engine.analyze_results(results)
        
        assert "winner" in analysis
        assert "metrics" in analysis
        assert analysis["metrics"]["fastest_provider"] == "provider1"
        assert analysis["metrics"]["cheapest_provider"] == "provider1"
        assert analysis["metrics"]["success_rate"] == 1.0


class TestConstraints:
    """Test constraint suggestion system."""
    
    def test_analyze_prompt(self):
        """Test prompt type analysis."""
        from akab.constraints import ConstraintSuggester
        
        suggester = ConstraintSuggester(Mock())
        
        assert suggester.analyze_prompt("Write a function to sort a list") == "code"
        assert suggester.analyze_prompt("Analyze the market trends") == "analytical"
        assert suggester.analyze_prompt("Write a creative story") == "creative"
        assert suggester.analyze_prompt("Let's have a conversation") == "conversation"
        assert suggester.analyze_prompt("Explain quantum physics") == "general"
        
    def test_get_default_constraints(self):
        """Test getting default constraints."""
        from akab.constraints import ConstraintSuggester
        
        suggester = ConstraintSuggester(Mock())
        
        constraints = suggester.get_default_constraints("code")
        assert constraints["temperature"] < 0.5
        assert constraints["max_tokens"] >= 1500
        
    def test_validate_constraints(self):
        """Test constraint validation."""
        from akab.constraints import ConstraintSuggester
        
        suggester = ConstraintSuggester(Mock())
        
        # Valid constraints
        valid, issues = suggester.validate_constraints(
            {"max_tokens": 1000, "temperature": 0.7},
            ["anthropic_m"]
        )
        assert valid is True
        assert len(issues) == 0
        
        # Invalid constraints
        invalid, issues = suggester.validate_constraints(
            {"max_tokens": -100, "temperature": 3.0},
            ["anthropic_m"]
        )
        assert invalid is False
        assert len(issues) > 0
