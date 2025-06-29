"""Local storage for AKAB campaigns and results."""

import json
import os
import aiofiles
from pathlib import Path
from typing import Any, Dict, Optional

# Import substrate components
try:
    from substrate import Campaign
except ImportError:
    try:
        from .substrate import Campaign
    except ImportError:
        from .substrate_stub import Campaign


class LocalStorage:
    """Simple file-based storage for AKAB data."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize local storage.
        
        Args:
            base_path: Base directory for storage (defaults to AKAB_DATA_PATH env var)
        """
        if base_path is None:
            base_path = os.getenv("AKAB_DATA_PATH", "./akab_data")
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.campaigns_dir = self.base_path / "campaigns"
        self.prompts_dir = self.base_path / "prompts"
        self.results_dir = self.base_path / "results"
        
        for dir_path in [self.campaigns_dir, self.prompts_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
            
    async def save_campaign(self, campaign: Campaign) -> None:
        """Save campaign to storage.
        
        Args:
            campaign: Campaign to save
        """
        file_path = self.campaigns_dir / f"{campaign.id}.json"
        
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(campaign.dict(), indent=2))
            
    async def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get campaign by ID.
        
        Args:
            campaign_id: Campaign ID
            
        Returns:
            Campaign if found, None otherwise
        """
        file_path = self.campaigns_dir / f"{campaign_id}.json"
        
        if not file_path.exists():
            return None
            
        async with aiofiles.open(file_path, "r") as f:
            data = json.loads(await f.read())
            return Campaign(**data)
            
    async def save_prompt(self, prompt_id: str, prompt_data: Dict[str, Any]) -> None:
        """Save prompt data.
        
        Args:
            prompt_id: Prompt ID
            prompt_data: Prompt data to save
        """
        file_path = self.prompts_dir / f"{prompt_id}.json"
        
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(prompt_data, indent=2))
            
    async def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get prompt by ID.
        
        Args:
            prompt_id: Prompt ID
            
        Returns:
            Prompt data if found, None otherwise
        """
        file_path = self.prompts_dir / f"{prompt_id}.json"
        
        if not file_path.exists():
            return None
            
        async with aiofiles.open(file_path, "r") as f:
            return json.loads(await f.read())
            
    async def save_results(self, campaign_id: str, results: Dict[str, Any]) -> None:
        """Save campaign results.
        
        Args:
            campaign_id: Campaign ID
            results: Results to save
        """
        file_path = self.results_dir / f"{campaign_id}_results.json"
        
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(results, indent=2))
            
    async def get_results(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get campaign results.
        
        Args:
            campaign_id: Campaign ID
            
        Returns:
            Results if found, None otherwise
        """
        file_path = self.results_dir / f"{campaign_id}_results.json"
        
        if not file_path.exists():
            return None
            
        async with aiofiles.open(file_path, "r") as f:
            return json.loads(await f.read())
            
    async def list_campaigns(self, status: Optional[str] = None) -> list[str]:
        """List campaign IDs, optionally filtered by status.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of campaign IDs
        """
        campaign_ids = []
        
        for file_path in self.campaigns_dir.glob("*.json"):
            if status:
                # Load campaign to check status
                async with aiofiles.open(file_path, "r") as f:
                    data = json.loads(await f.read())
                    if data.get("status") == status:
                        campaign_ids.append(file_path.stem)
            else:
                campaign_ids.append(file_path.stem)
                
        return campaign_ids
        
    async def delete_campaign(self, campaign_id: str) -> bool:
        """Delete a campaign and its associated data.
        
        Args:
            campaign_id: Campaign ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        campaign_file = self.campaigns_dir / f"{campaign_id}.json"
        results_file = self.results_dir / f"{campaign_id}_results.json"
        
        deleted = False
        
        if campaign_file.exists():
            campaign_file.unlink()
            deleted = True
            
        if results_file.exists():
            results_file.unlink()
            
        # Also delete associated prompts
        # (In production, would track prompt usage to avoid deleting shared prompts)
        
        return deleted
        
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Storage statistics
        """
        stats = {
            "campaigns": len(list(self.campaigns_dir.glob("*.json"))),
            "prompts": len(list(self.prompts_dir.glob("*.json"))),
            "results": len(list(self.results_dir.glob("*.json"))),
            "total_size_bytes": 0
        }
        
        # Calculate total size
        for dir_path in [self.campaigns_dir, self.prompts_dir, self.results_dir]:
            for file_path in dir_path.glob("*.json"):
                stats["total_size_bytes"] += file_path.stat().st_size
                
        return stats
