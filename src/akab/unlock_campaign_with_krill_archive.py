"""Updated unlock_campaign implementation with direct krill archiving

This replaces the existing unlock_campaign method in AkabServer class.
"""

async def unlock_campaign(self, campaign_id: str) -> Dict[str, Any]:
    """Unlock Level 2 campaign to reveal blinding mappings with krill archiving"""
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
        
        # Create archive in krill
        try:
            import shutil
            from pathlib import Path
            
            # Create campaign archive directory in krill
            krill_archive_dir = Path(self.krill_dir) / "archive" / campaign_id
            krill_archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Get campaign directory path
            campaign_level_dir = {
                1: "quick",
                2: "standard", 
                3: "experiments"
            }.get(campaign_level, "standard")
            
            campaign_source_dir = Path(self.data_dir) / "campaigns" / campaign_level_dir
            campaign_file = campaign_source_dir / f"{campaign_id}.json"
            
            # Copy blinded state (before unlock)
            blinded_dir = krill_archive_dir / "blinded"
            blinded_dir.mkdir(exist_ok=True)
            
            if campaign_file.exists():
                shutil.copy2(campaign_file, blinded_dir / "campaign.json")
            
            # Also copy results if they exist
            results_dir = Path(self.data_dir) / "results" / campaign_id
            if results_dir.exists():
                shutil.copytree(results_dir, blinded_dir / "results", dirs_exist_ok=True)
            
            # Now unlock the campaign
            campaign.metadata["unlocked"] = True
            campaign.metadata["unlocked_at"] = time.time()
            campaign.metadata["blinding_map"] = blinding_map
            
            # Save the updated campaign
            await self.campaign_manager._save_campaign(campaign)
            
            # Copy clear state (after unlock)
            clear_dir = krill_archive_dir / "clear"
            clear_dir.mkdir(exist_ok=True)
            
            # Copy the now-unlocked campaign file
            shutil.copy2(campaign_file, clear_dir / "campaign.json")
            
            # Copy results again for clear version
            if results_dir.exists():
                shutil.copytree(results_dir, clear_dir / "results", dirs_exist_ok=True)
            
            # Create archive metadata
            archive_metadata = {
                "campaign_id": campaign_id,
                "campaign_name": campaign.name,
                "level": campaign_level,
                "unlocked_at": campaign.metadata["unlocked_at"],
                "blinding_map": blinding_map,
                "archive_created": time.time(),
                "paths": {
                    "blinded": str(blinded_dir),
                    "clear": str(clear_dir)
                }
            }
            
            # Save archive metadata
            with open(krill_archive_dir / "metadata.json", 'w') as f:
                json.dump(archive_metadata, f, indent=2)
            
        except Exception as archive_error:
            # Log but don't fail the unlock
            print(f"Warning: Krill archive failed: {archive_error}")
            archive_metadata = {"error": str(archive_error)}
        
        # Create provider mapping details for response
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
        
        # Add archive info
        if "error" not in archive_metadata:
            response_data["archive"] = {
                "status": "success",
                "location": f"/krill/archive/{campaign_id}",
                "created_at": archive_metadata["archive_created"]
            }
        else:
            response_data["archive"] = {
                "status": "failed",
                "error": archive_metadata["error"]
            }
        
        return self.create_response(
            data=response_data,
            message="Campaign unlocked! Provider mappings revealed and archived to krill."
        )
        
    except ValidationError:
        raise
    except Exception as e:
        return self.create_error_response(str(e))
