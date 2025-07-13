"""Campaign management for A/B testing"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import aiofiles
from dataclasses import dataclass, asdict


@dataclass
class Campaign:
    """A/B testing campaign"""
    id: str
    name: str
    description: str
    variants: List[Dict[str, Any]]  # List of variant configurations
    created_at: float
    status: str = "active"  # active, completed, cancelled
    results: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    success_criteria: Dict[str, Any] = None  # Level 2: Success criteria
    level: int = 2  # Campaign level (1=quick, 2=standard, 3=experiment)
    
    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Campaign":
        """Create from dictionary"""
        return cls(**data)


class CampaignManager:
    """Manages A/B testing campaigns"""
    
    def __init__(self, data_dir: str, krill_dir: str):
        self.data_dir = data_dir
        # Store everything in krill for proper isolation
        self.campaigns_dir = os.path.join(krill_dir, "campaigns")
        self.results_dir = os.path.join(krill_dir, "results")
        self.krill_results_dir = os.path.join(krill_dir, "results")  # Same as results_dir now
        
        # Ensure directories exist
        os.makedirs(self.campaigns_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def create_campaign(self, name: str, description: str, 
                            variants: List[Dict[str, Any]],
                            success_criteria: Dict[str, Any] = None,
                            level: int = 2) -> Campaign:
        """Create new campaign with optional success criteria"""
        # Generate ID
        campaign_id = f"campaign_{int(time.time() * 1000)}"
        
        # Default success criteria if not provided
        if success_criteria is None and level == 2:
            success_criteria = {
                "primary_metric": "execution_time",
                "direction": "minimize",
                "evaluation_method": "statistical"
            }
        
        # Create campaign
        campaign = Campaign(
            id=campaign_id,
            name=name,
            description=description,
            variants=variants,
            created_at=time.time(),
            success_criteria=success_criteria,
            level=level
        )
        
        # Save to appropriate directory based on level
        await self._save_campaign(campaign)
        
        return campaign
    
    async def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get campaign by ID from any level directory"""
        # Search in all level directories
        for level_dir in ["quick", "standard", "experiments"]:
            filepath = os.path.join(self.campaigns_dir, level_dir, f"{campaign_id}.json")
            
            if os.path.exists(filepath):
                async with aiofiles.open(filepath, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    return Campaign.from_dict(data)
        
        # Also check old location for backward compatibility
        old_filepath = os.path.join(self.campaigns_dir, f"{campaign_id}.json")
        if os.path.exists(old_filepath):
            async with aiofiles.open(old_filepath, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                return Campaign.from_dict(data)
        
        return None
    
    async def list_campaigns(self, status: Optional[str] = None) -> List[Campaign]:
        """List all campaigns from all level directories"""
        campaigns = []
        
        # Search in all level directories
        for level_dir in ["quick", "standard", "experiments", ""]:
            if level_dir:
                search_dir = os.path.join(self.campaigns_dir, level_dir)
            else:
                search_dir = self.campaigns_dir  # Root for backward compatibility
                
            if os.path.exists(search_dir):
                for filename in os.listdir(search_dir):
                    if filename.endswith('.json') and not filename.startswith('.'):
                        filepath = os.path.join(search_dir, filename)
                        try:
                            async with aiofiles.open(filepath, 'r') as f:
                                content = await f.read()
                                data = json.loads(content)
                                campaign = Campaign.from_dict(data)
                                
                                if status is None or campaign.status == status:
                                    campaigns.append(campaign)
                        except Exception as e:
                            # Skip corrupted files
                            continue
        
        # Sort by created_at descending
        campaigns.sort(key=lambda c: c.created_at, reverse=True)
        
        return campaigns
    
    async def add_result(self, campaign_id: str, result: Dict[str, Any]) -> bool:
        """Add result to campaign"""
        campaign = await self.get_campaign(campaign_id)
        
        if not campaign:
            return False
        
        # Add timestamp if not present
        if "timestamp" not in result:
            result["timestamp"] = time.time()
        
        # Add to results
        campaign.results.append(result)
        
        # Save campaign
        await self._save_campaign(campaign)
        
        # Also save individual result
        await self._save_result(campaign_id, result)
        
        return True
    
    async def complete_campaign(self, campaign_id: str) -> bool:
        """Mark campaign as completed"""
        campaign = await self.get_campaign(campaign_id)
        
        if not campaign:
            return False
        
        campaign.status = "completed"
        campaign.metadata["completed_at"] = time.time()
        
        # Save campaign
        await self._save_campaign(campaign)
        
        # Archive to krill
        await self._archive_to_krill(campaign)
        
        return True
    
    async def get_campaign_results(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Get all results for a campaign"""
        campaign = await self.get_campaign(campaign_id)
        
        if not campaign:
            return []
        
        return campaign.results
    
    async def _save_campaign(self, campaign: Campaign):
        """Save campaign to file in level-appropriate directory"""
        # Determine directory based on level
        if campaign.level == 1:
            level_dir = os.path.join(self.campaigns_dir, "quick")
        elif campaign.level == 2:
            level_dir = os.path.join(self.campaigns_dir, "standard")
        else:  # Level 3
            level_dir = os.path.join(self.campaigns_dir, "experiments")
        
        # Ensure level directory exists
        os.makedirs(level_dir, exist_ok=True)
        
        # Save campaign
        filepath = os.path.join(level_dir, f"{campaign.id}.json")
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(campaign.to_dict(), indent=2))
    
    async def _save_result(self, campaign_id: str, result: Dict[str, Any]):
        """Save individual result"""
        # Create campaign results directory
        campaign_results_dir = os.path.join(self.results_dir, campaign_id)
        os.makedirs(campaign_results_dir, exist_ok=True)
        
        # Generate result ID
        result_id = f"result_{int(time.time() * 1000000)}"
        filepath = os.path.join(campaign_results_dir, f"{result_id}.json")
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(result, indent=2))
    
    async def _archive_to_krill(self, campaign: Campaign):
        """Archive completed campaign to krill"""
        # Create archive with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{campaign.id}_{timestamp}.json"
        
        # Prepare archive data
        archive_data = {
            "campaign": campaign.to_dict(),
            "archived_at": time.time(),
            "archive_type": "akab_campaign"
        }
        
        # Save to krill results
        filepath = os.path.join(self.krill_results_dir, archive_name)
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(archive_data, indent=2))
