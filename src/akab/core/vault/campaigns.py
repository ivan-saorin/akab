"""Campaign storage and management for AKAB"""
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
import aiofiles


@dataclass
class Campaign:
    """A/B testing campaign data structure"""
    id: str
    name: str
    description: str
    variants: List[Dict[str, Any]]  # List of variant configurations
    created_at: float
    status: str = "created"  # created, running, completed, cancelled, archived
    results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    level: int = 2  # 1=quick, 2=campaign, 3=experiment
    variant_mapping: Dict[str, str] = field(default_factory=dict)  # For blinding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Campaign":
        """Create from dictionary"""
        return cls(**data)
    
    def add_result(self, result: Dict[str, Any]):
        """Add execution result"""
        result["timestamp"] = time.time()
        self.results.append(result)
    
    def get_variant_by_id(self, variant_id: str) -> Optional[Dict[str, Any]]:
        """Get variant configuration by ID"""
        for variant in self.variants:
            if variant["id"] == variant_id:
                return variant
        return None


class CampaignVault:
    """AKAB's campaign storage system
    
    Uses /krill directory for Level 3 experiments to ensure
    complete isolation from LLM access.
    """
    
    def __init__(self, base_path: str = "/krill"):
        self.base_path = Path(base_path)
        
        # Directory structure
        self.campaigns_dir = self.base_path / "campaigns"
        self.experiments_dir = self.base_path / "experiments"
        self.archives_dir = self.base_path / "archives"
        self.mappings_dir = self.base_path / "mappings"
        
        # Create directories
        for dir_path in [self.campaigns_dir, self.experiments_dir, 
                        self.archives_dir, self.mappings_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def store_campaign(self, campaign: Campaign) -> str:
        """Store campaign based on its level"""
        # Determine storage location based on level
        if campaign.level == 3:
            # Level 3 experiments go in isolated directory
            filepath = self.experiments_dir / f"{campaign.id}.json"
        else:
            # Level 1 (quick) and Level 2 (campaign) in regular directory
            filepath = self.campaigns_dir / f"{campaign.id}.json"
        
        # Save campaign data
        data = campaign.to_dict()
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(data, indent=2))
        
        # Store mapping for Level 2/3 (blinded)
        if campaign.level >= 2 and campaign.variant_mapping:
            mapping_file = self.mappings_dir / f"{campaign.id}_mapping.json"
            async with aiofiles.open(mapping_file, 'w') as f:
                await f.write(json.dumps(campaign.variant_mapping, indent=2))
        
        return str(filepath)
    
    async def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Retrieve campaign by ID"""
        # Search in both locations
        for directory in [self.campaigns_dir, self.experiments_dir]:
            filepath = directory / f"{campaign_id}.json"
            
            if filepath.exists():
                async with aiofiles.open(filepath, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    # Load mapping if exists
                    mapping_file = self.mappings_dir / f"{campaign_id}_mapping.json"
                    if mapping_file.exists():
                        async with aiofiles.open(mapping_file, 'r') as mf:
                            mapping_content = await mf.read()
                            data["variant_mapping"] = json.loads(mapping_content)
                    
                    return Campaign.from_dict(data)
        
        return None
    
    async def update_campaign(self, campaign: Campaign) -> bool:
        """Update existing campaign"""
        # Find where it's stored
        if campaign.level == 3:
            filepath = self.experiments_dir / f"{campaign.id}.json"
        else:
            filepath = self.campaigns_dir / f"{campaign.id}.json"
        
        if not filepath.exists():
            return False
        
        # Update the file
        data = campaign.to_dict()
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(data, indent=2))
        
        return True
    
    async def list_campaigns(self, level: Optional[int] = None, 
                           status: Optional[str] = None) -> List[Campaign]:
        """List campaigns with optional filters"""
        campaigns = []
        
        # Determine which directories to search
        if level == 3:
            directories = [self.experiments_dir]
        elif level in [1, 2]:
            directories = [self.campaigns_dir]
        else:
            directories = [self.campaigns_dir, self.experiments_dir]
        
        # Load campaigns from directories
        for directory in directories:
            for filepath in directory.glob("*.json"):
                # Skip mapping files
                if "_mapping" in filepath.name:
                    continue
                
                try:
                    async with aiofiles.open(filepath, 'r') as f:
                        content = await f.read()
                        data = json.loads(content)
                        campaign = Campaign.from_dict(data)
                        
                        # Apply status filter if provided
                        if status and campaign.status != status:
                            continue
                        
                        campaigns.append(campaign)
                except Exception as e:
                    # Log error but continue
                    print(f"Error loading campaign {filepath}: {e}")
        
        # Sort by creation time (newest first)
        campaigns.sort(key=lambda c: c.created_at, reverse=True)
        
        return campaigns
    
    async def archive_campaign(self, campaign_id: str) -> Optional[str]:
        """Archive a completed campaign"""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            return None
        
        # Update status
        campaign.status = "archived"
        campaign.metadata["archived_at"] = time.time()
        
        # Move to archives directory
        archive_path = self.archives_dir / f"{campaign_id}_{int(time.time())}.json"
        
        # Save to archive
        data = campaign.to_dict()
        async with aiofiles.open(archive_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))
        
        # Delete from original location
        if campaign.level == 3:
            original_path = self.experiments_dir / f"{campaign_id}.json"
        else:
            original_path = self.campaigns_dir / f"{campaign_id}.json"
        
        if original_path.exists():
            original_path.unlink()
        
        # Also archive mapping if exists
        mapping_file = self.mappings_dir / f"{campaign_id}_mapping.json"
        if mapping_file.exists():
            archive_mapping = self.archives_dir / f"{campaign_id}_mapping_{int(time.time())}.json"
            mapping_file.rename(archive_mapping)
        
        return str(archive_path)
    
    async def get_variant_mapping(self, campaign_id: str) -> Optional[Dict[str, str]]:
        """Get variant mapping for a campaign (if unlocked)"""
        mapping_file = self.mappings_dir / f"{campaign_id}_mapping.json"
        
        if mapping_file.exists():
            async with aiofiles.open(mapping_file, 'r') as f:
                content = await f.read()
                return json.loads(content)
        
        # Check archives
        for archive_file in self.archives_dir.glob(f"{campaign_id}_mapping_*.json"):
            async with aiofiles.open(archive_file, 'r') as f:
                content = await f.read()
                return json.loads(content)
        
        return None
