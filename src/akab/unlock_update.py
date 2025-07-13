"""Updated unlock_campaign implementation with archiving

Add this to the AkabServer class to replace the existing unlock_campaign method.
"""

async def unlock_campaign(self, campaign_id: str) -> Dict[str, Any]:
    """Unlock Level 2 campaign to reveal blinding mappings with archiving"""
    try:
        campaign = await self.campaign_manager.get_campaign(campaign_id)
        
        if not campaign:
            raise ValidationError(
                f"Campaign not found: {campaign_id}",
                field="campaign_id"
            )
        
        # Default level to 2 for backward compatibility
        campaign_level = getattr(campaign, 'level', 2)
        
        if campaign_level != 2:
            return self.create_response(
                data={"error": f"Campaign is Level {campaign_level}, unlock only applies to Level 2"},
                success=False
            )
        
        # Check if already unlocked
        if hasattr(campaign, 'metadata') and campaign.metadata.get('unlocked', False):
            return self.create_response(
                data={"error": "Campaign is already unlocked"},
                success=False
            )
        
        # Get blinding mappings before unlocking
        variant_ids = [v["id"] for v in campaign.variants]
        blinding_map = self.laboratory.create_blind_mapping(variant_ids, campaign_id)
        
        # Initialize archive manager if not exists
        if not hasattr(self, 'archive_manager'):
            from .archive_manager import ArchiveManager
            self.archive_manager = ArchiveManager(self.data_dir)
        
        # Archive the campaign state
        archive_result = await self.archive_manager.archive_campaign_unlock(
            campaign_id=campaign_id,
            campaign_data=campaign.to_dict(),
            blinding_map=blinding_map
        )
        
        # Check if archiving failed
        if "error" in archive_result:
            # Log warning but continue with unlock
            print(f"Warning: Archive failed: {archive_result['error']}")
            # Don't fail the unlock operation
        
        # Mark campaign as unlocked
        campaign.metadata["unlocked"] = True
        campaign.metadata["unlocked_at"] = time.time()
        campaign.metadata["archive_info"] = archive_result if "error" not in archive_result else None
        
        # Save the updated campaign
        await self.campaign_manager._save_campaign(campaign)
        
        # Create provider mapping details
        mappings = {}
        for variant in campaign.variants:
            variant_id = variant["id"]
            blinded_id = blinding_map[variant_id]
            mappings[variant_id] = {
                "blinded_id": blinded_id,
                "provider": variant["provider"],
                "model": variant["model"],
                "prompt": variant["prompt"][:50] + "..." if len(variant["prompt"]) > 50 else variant["prompt"]
            }
        
        response_data = {
            "campaign_id": campaign_id,
            "campaign_name": campaign.name,
            "level": campaign_level,
            "mappings": mappings,
            "blinding_map": blinding_map,
            "unlocked_at": campaign.metadata["unlocked_at"]
        }
        
        # Add archive info if successful
        if "error" not in archive_result:
            response_data["archive"] = {
                "status": "success",
                "paths": archive_result["archive_paths"],
                "timestamp": archive_result["unlock_timestamp"]
            }
        else:
            response_data["archive"] = {
                "status": "failed",
                "error": archive_result["error"]
            }
        
        return self.create_response(
            data=response_data,
            message="Campaign unlocked! Provider mappings revealed and archived."
        )
        
    except ValidationError:
        raise
    except Exception as e:
        return self.create_error_response(str(e))
