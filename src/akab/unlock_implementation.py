"""Complete unlock implementation for akab

This file contains the full unlock method and helper methods to replace
the existing implementation in server.py
"""

async def unlock(self, id: str) -> Dict[str, Any]:
    """Unlock campaign or experiment to reveal mappings with krill archiving"""
    try:
        # Try to get as campaign first
        campaign = await self.campaign_manager.get_campaign(id)
        
        if not campaign:
            raise ValidationError(
                f"Campaign or experiment not found: {id}",
                field="id"
            )
        
        # Get campaign level and check if it's an experiment
        campaign_level = getattr(campaign, 'level', 2)
        is_experiment = campaign.metadata.get("is_experiment", False)
        
        # Handle based on type
        if campaign_level == 3 or is_experiment:
            # This is a Level 3 experiment
            return await self._unlock_experiment(campaign)
        elif campaign_level == 2:
            # This is a regular Level 2 campaign
            return await self._unlock_campaign(campaign)
        else:
            # Level 1 campaigns don't need unlocking
            return self.create_response(
                data={"error": f"Level {campaign_level} campaigns don't support unlocking"},
                success=False
            )
        
    except ValidationError:
        raise
    except Exception as e:
        return self.create_error_response(str(e))

async def _unlock_campaign(self, campaign: Campaign) -> Dict[str, Any]:
    """Unlock a Level 2 campaign"""
    campaign_id = campaign.id
    
    # Check if already unlocked
    if campaign.metadata.get('unlocked', False):
        return self.create_response(
            data={"error": "Campaign is already unlocked"},
            success=False
        )
    
    # Get blinding mappings before unlocking
    variant_ids = [v["id"] for v in campaign.variants]
    blinding_map = self.laboratory.create_blind_mapping(variant_ids, campaign_id)
    
    # Create archive in krill
    archive_metadata = {}
    try:
        import shutil
        from pathlib import Path
        
        # Create campaign archive directory in krill
        krill_archive_dir = Path(self.krill_dir) / "archive" / campaign_id
        krill_archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Get campaign directory path
        campaign_source_dir = Path(self.data_dir) / "campaigns" / "standard"
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
            "level": 2,
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
        logger.warning(f"Krill archive failed: {archive_error}", exc_info=True)
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
        "id": campaign_id,
        "name": campaign.name,
        "type": "campaign",
        "level": 2,
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

async def _unlock_experiment(self, experiment: Campaign) -> Dict[str, Any]:
    """Unlock a Level 3 experiment (only if complete)"""
    experiment_id = experiment.id
    
    # Check if already revealed/unlocked
    if experiment.metadata.get("revealed", False):
        # Experiment was revealed, now unlock it
        return await self._perform_experiment_unlock(experiment)
    else:
        # Experiment not yet revealed
        success_criteria = experiment.success_criteria or {}
        min_iterations = success_criteria.get("minimum_iterations", 30)
        total_tests = len(experiment.results)
        
        if total_tests < min_iterations:
            return self.create_error_response(
                f"Cannot unlock ongoing experiment: {total_tests}/{min_iterations} tests completed",
                suggestions=[
                    f"Complete {min_iterations - total_tests} more tests",
                    "Use akab_reveal_experiment when ready",
                    "Use akab_diagnose_experiment to check status"
                ]
            )
        else:
            return self.create_error_response(
                "Experiment complete but not revealed. Use akab_reveal_experiment first.",
                suggestions=[
                    "Run akab_reveal_experiment to check statistical significance",
                    "If no significance found, use akab_diagnose_experiment"
                ]
            )

async def _perform_experiment_unlock(self, experiment: Campaign) -> Dict[str, Any]:
    """Actually unlock and archive a revealed experiment"""
    experiment_id = experiment.id
    
    # Check if already unlocked
    if experiment.metadata.get("unlocked", False):
        return self.create_response(
            data={"error": "Experiment is already unlocked"},
            success=False
        )
    
    # Get all the mappings (from reveal and scrambling)
    revealed_mappings = experiment.metadata.get("revealed_mappings", {})
    winner = experiment.metadata.get("winner")
    
    # Create archive in krill
    archive_metadata = {}
    try:
        import shutil
        from pathlib import Path
        
        # Create experiment archive directory in krill
        krill_archive_dir = Path(self.krill_dir) / "archive" / experiment_id
        krill_archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Get experiment directory path
        experiment_source_dir = Path(self.data_dir) / "campaigns" / "experiments"
        experiment_file = experiment_source_dir / f"{experiment_id}.json"
        
        # Copy blinded state (before adding full mappings)
        blinded_dir = krill_archive_dir / "blinded"
        blinded_dir.mkdir(exist_ok=True)
        
        # Create a blinded version without revealed mappings
        blinded_experiment = experiment.to_dict()
        blinded_experiment["metadata"].pop("revealed_mappings", None)
        blinded_experiment["metadata"].pop("winner", None)
        
        with open(blinded_dir / "experiment.json", 'w') as f:
            json.dump(blinded_experiment, f, indent=2)
        
        # Copy results
        results_dir = Path(self.data_dir) / "results" / experiment_id
        if results_dir.exists():
            shutil.copytree(results_dir, blinded_dir / "results", dirs_exist_ok=True)
        
        # Now unlock the experiment
        experiment.metadata["unlocked"] = True
        experiment.metadata["unlocked_at"] = time.time()
        
        # Save the updated experiment
        await self.campaign_manager._save_campaign(experiment)
        
        # Copy clear state (with all mappings)
        clear_dir = krill_archive_dir / "clear"
        clear_dir.mkdir(exist_ok=True)
        
        # Copy the full experiment with mappings
        if experiment_file.exists():
            shutil.copy2(experiment_file, clear_dir / "experiment.json")
        
        # Copy results again
        if results_dir.exists():
            shutil.copytree(results_dir, clear_dir / "results", dirs_exist_ok=True)
        
        # Create comprehensive archive metadata
        archive_metadata = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "level": 3,
            "hypothesis": experiment.metadata.get("hypothesis"),
            "revealed_at": experiment.metadata.get("revealed_at"),
            "unlocked_at": experiment.metadata["unlocked_at"],
            "winner": winner,
            "revealed_mappings": revealed_mappings,
            "total_tests": len(experiment.results),
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
        logger.warning(f"Krill archive failed: {archive_error}", exc_info=True)
        archive_metadata = {"error": str(archive_error)}
    
    response_data = {
        "id": experiment_id,
        "name": experiment.name,
        "type": "experiment",
        "level": 3,
        "hypothesis": experiment.metadata.get("hypothesis"),
        "winner": winner,
        "revealed_mappings": revealed_mappings,
        "unlocked_at": experiment.metadata["unlocked_at"]
    }
    
    # Add archive info
    if "error" not in archive_metadata:
        response_data["archive"] = {
            "status": "success",
            "location": f"/krill/archive/{experiment_id}",
            "created_at": archive_metadata["archive_created"]
        }
    else:
        response_data["archive"] = {
            "status": "failed",
            "error": archive_metadata["error"]
        }
    
    return self.create_response(
        data=response_data,
        message="Experiment unlocked! All mappings revealed and archived to krill."
    )
