"""Archive management for completed campaigns and experiments"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import aiofiles


class ArchiveManager:
    """Manages archived campaigns and experiments
    
    Provides access to historical data while keeping
    active campaigns separate.
    """
    
    def __init__(self, base_path: str = "/krill"):
        self.base_path = Path(base_path)
        self.archives_dir = self.base_path / "archives"
        self.archives_dir.mkdir(parents=True, exist_ok=True)
    
    async def list_archives(self, 
                          campaign_type: Optional[str] = None,
                          date_from: Optional[float] = None,
                          date_to: Optional[float] = None) -> List[Dict[str, Any]]:
        """List archived campaigns with optional filters"""
        archives = []
        
        for filepath in self.archives_dir.glob("*.json"):
            # Skip mapping files
            if "_mapping" in filepath.name:
                continue
            
            try:
                # Extract timestamp from filename
                parts = filepath.stem.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    archive_time = int(parts[-1])
                else:
                    archive_time = filepath.stat().st_mtime
                
                # Apply date filters
                if date_from and archive_time < date_from:
                    continue
                if date_to and archive_time > date_to:
                    continue
                
                # Load archive metadata
                async with aiofiles.open(filepath, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    # Apply type filter
                    if campaign_type:
                        if campaign_type == "experiment" and data.get("level") != 3:
                            continue
                        elif campaign_type == "campaign" and data.get("level") != 2:
                            continue
                        elif campaign_type == "quick" and data.get("level") != 1:
                            continue
                    
                    archives.append({
                        "filepath": str(filepath),
                        "campaign_id": data.get("id"),
                        "name": data.get("name"),
                        "level": data.get("level"),
                        "archived_at": archive_time,
                        "created_at": data.get("created_at"),
                        "status": data.get("status"),
                        "variants_count": len(data.get("variants", [])),
                        "results_count": len(data.get("results", []))
                    })
            
            except Exception as e:
                print(f"Error loading archive {filepath}: {e}")
        
        # Sort by archive time (newest first)
        archives.sort(key=lambda a: a["archived_at"], reverse=True)
        
        return archives
    
    async def get_archive(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get full archived campaign data"""
        # Search for the campaign in archives
        for filepath in self.archives_dir.glob(f"{campaign_id}_*.json"):
            if "_mapping" not in filepath.name:
                async with aiofiles.open(filepath, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    # Also load mapping if available
                    mapping_pattern = f"{campaign_id}_mapping_*.json"
                    for mapping_file in self.archives_dir.glob(mapping_pattern):
                        async with aiofiles.open(mapping_file, 'r') as mf:
                            mapping_content = await mf.read()
                            data["variant_mapping"] = json.loads(mapping_content)
                        break
                    
                    return data
        
        return None
    
    async def export_archive(self, campaign_id: str, 
                           export_format: str = "json") -> Optional[str]:
        """Export archived campaign in specified format"""
        data = await self.get_archive(campaign_id)
        if not data:
            return None
        
        # Generate export filename
        timestamp = int(time.time())
        export_dir = self.base_path / "exports"
        export_dir.mkdir(exist_ok=True)
        
        if export_format == "json":
            filepath = export_dir / f"{campaign_id}_export_{timestamp}.json"
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        
        elif export_format == "summary":
            # Create human-readable summary
            filepath = export_dir / f"{campaign_id}_summary_{timestamp}.txt"
            summary = self._create_summary(data)
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(summary)
        
        else:
            return None
        
        return str(filepath)
    
    def _create_summary(self, data: Dict[str, Any]) -> str:
        """Create human-readable summary of campaign"""
        lines = [
            f"Campaign Summary: {data.get('name', 'Unnamed')}",
            f"ID: {data.get('id')}",
            f"Level: {data.get('level')} ({'Quick' if data.get('level') == 1 else 'Campaign' if data.get('level') == 2 else 'Experiment'})",
            f"Status: {data.get('status')}",
            f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data.get('created_at', 0)))}",
            f"Description: {data.get('description', 'No description')}",
            "",
            f"Variants: {len(data.get('variants', []))}",
            f"Results: {len(data.get('results', []))}",
            ""
        ]
        
        # Add variant information
        if data.get('variant_mapping'):
            lines.append("Variant Mapping (Unlocked):")
            for variant_id, model_id in data['variant_mapping'].items():
                lines.append(f"  {variant_id} -> {model_id}")
            lines.append("")
        
        # Add success criteria
        if data.get('success_criteria'):
            lines.append("Success Criteria:")
            for key, value in data['success_criteria'].items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Add results summary
        if data.get('results'):
            lines.append("Results Summary:")
            total_results = len(data['results'])
            lines.append(f"  Total executions: {total_results}")
            
            # Group by variant
            variant_counts = {}
            for result in data['results']:
                variant = result.get('variant_id', 'unknown')
                variant_counts[variant] = variant_counts.get(variant, 0) + 1
            
            for variant, count in variant_counts.items():
                lines.append(f"  {variant}: {count} executions")
        
        return "\n".join(lines)
