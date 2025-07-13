"""Archive Manager for Campaign Unlocking

This module handles archiving campaigns when they're unlocked.
Instead of directly writing to krill (which is outside allowed paths),
we create archives within the allowed data directory and log the action.
"""

import os
import json
import shutil
import asyncio
import aiofiles
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class ArchiveManager:
    """Manages campaign archiving for unlock operations"""
    
    def __init__(self, data_dir: str, archive_dir: Optional[str] = None):
        """
        Initialize archive manager
        
        Args:
            data_dir: Base data directory (allowed path)
            archive_dir: Optional archive directory (defaults to data_dir/archives)
        """
        self.data_dir = Path(data_dir)
        self.archive_dir = Path(archive_dir) if archive_dir else self.data_dir / "archives"
        
        # Ensure archive directory exists
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    async def archive_campaign_unlock(self, campaign_id: str, campaign_data: Dict[str, Any],
                                    blinding_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Archive campaign state before and after unlocking
        
        Args:
            campaign_id: Campaign identifier
            campaign_data: Current campaign data
            blinding_map: Variant ID to blinded ID mapping
            
        Returns:
            Archive metadata including paths
        """
        try:
            # Create campaign-specific archive directory
            campaign_archive_dir = self.archive_dir / campaign_id
            campaign_archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped subdirectory for this unlock event
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unlock_dir = campaign_archive_dir / f"unlock_{timestamp}"
            unlock_dir.mkdir(exist_ok=True)
            
            # Save blinded state
            blinded_dir = unlock_dir / "blinded"
            blinded_dir.mkdir(exist_ok=True)
            
            # Prepare blinded data (remove any provider info)
            blinded_data = self._prepare_blinded_data(campaign_data, blinding_map)
            
            # Write blinded state
            blinded_file = blinded_dir / "campaign.json"
            async with aiofiles.open(blinded_file, 'w') as f:
                await f.write(json.dumps(blinded_data, indent=2))
            
            # Save clear state
            clear_dir = unlock_dir / "clear"
            clear_dir.mkdir(exist_ok=True)
            
            # Prepare clear data (with full mappings)
            clear_data = self._prepare_clear_data(campaign_data, blinding_map)
            
            # Write clear state
            clear_file = clear_dir / "campaign.json"
            async with aiofiles.open(clear_file, 'w') as f:
                await f.write(json.dumps(clear_data, indent=2))
            
            # Create unlock metadata
            metadata = {
                "campaign_id": campaign_id,
                "unlock_timestamp": timestamp,
                "archive_paths": {
                    "root": str(unlock_dir),
                    "blinded": str(blinded_file),
                    "clear": str(clear_file)
                },
                "blinding_map": blinding_map,
                "stats": {
                    "total_results": len(campaign_data.get("results", [])),
                    "total_variants": len(campaign_data.get("variants", []))
                }
            }
            
            # Write metadata
            metadata_file = unlock_dir / "metadata.json"
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            # Log to krill manifest (just metadata, not actual data)
            await self._log_to_krill_manifest(campaign_id, metadata)
            
            return metadata
            
        except Exception as e:
            return {
                "error": f"Archive failed: {str(e)}",
                "campaign_id": campaign_id
            }
    
    def _prepare_blinded_data(self, campaign_data: Dict[str, Any], 
                             blinding_map: Dict[str, str]) -> Dict[str, Any]:
        """Prepare campaign data with blinding intact"""
        blinded = campaign_data.copy()
        
        # Remove provider information from variants
        if "variants" in blinded:
            blinded_variants = []
            for variant in blinded["variants"]:
                blinded_variant = {
                    "id": blinding_map.get(variant["id"], variant["id"]),
                    "constraints": variant.get("constraints", {})
                }
                blinded_variants.append(blinded_variant)
            blinded["variants"] = blinded_variants
        
        # Ensure results use blinded IDs
        if "results" in blinded:
            for result in blinded["results"]:
                if "provider" in result:
                    del result["provider"]
                if "model" in result:
                    del result["model"]
        
        return blinded
    
    def _prepare_clear_data(self, campaign_data: Dict[str, Any],
                           blinding_map: Dict[str, str]) -> Dict[str, Any]:
        """Prepare campaign data with clear mappings"""
        clear = campaign_data.copy()
        
        # Add mapping information
        clear["blinding_mappings"] = blinding_map
        clear["unlock_timestamp"] = datetime.now().isoformat()
        
        # Reverse map for results
        reverse_map = {v: k for k, v in blinding_map.items()}
        
        if "results" in clear:
            for result in clear["results"]:
                blinded_id = result.get("variant")
                if blinded_id and blinded_id in reverse_map:
                    result["original_variant_id"] = reverse_map[blinded_id]
        
        return clear
    
    async def _log_to_krill_manifest(self, campaign_id: str, metadata: Dict[str, Any]):
        """
        Log archive event to a manifest file (not actual krill directory)
        This maintains the audit trail without violating directory boundaries
        """
        manifest_file = self.archive_dir / "krill_manifest.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "campaign_unlock_archived",
            "campaign_id": campaign_id,
            "archive_location": metadata["archive_paths"]["root"],
            "note": "Archive created in allowed directory. Manual sync to krill required."
        }
        
        # Append to manifest
        async with aiofiles.open(manifest_file, 'a') as f:
            await f.write(json.dumps(log_entry) + "\n")
    
    async def list_archives(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """List available archives"""
        archives = []
        
        if campaign_id:
            # List archives for specific campaign
            campaign_dir = self.archive_dir / campaign_id
            if campaign_dir.exists():
                for unlock_dir in campaign_dir.glob("unlock_*"):
                    metadata_file = unlock_dir / "metadata.json"
                    if metadata_file.exists():
                        async with aiofiles.open(metadata_file, 'r') as f:
                            metadata = json.loads(await f.read())
                            archives.append(metadata)
        else:
            # List all campaign archives
            for campaign_dir in self.archive_dir.iterdir():
                if campaign_dir.is_dir() and campaign_dir.name != "krill_manifest.jsonl":
                    campaign_archives = await self.list_archives(campaign_dir.name)
                    archives.extend(campaign_archives["archives"])
        
        return {
            "archives": archives,
            "total": len(archives)
        }
