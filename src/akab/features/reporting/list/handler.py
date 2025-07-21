"""List Campaigns Handler"""
import time
from typing import Dict, Any, List, Optional
from datetime import datetime


class ListCampaignsHandler:
    """Handler for listing campaigns"""
    
    def __init__(self, response_builder, vault):
        self.response_builder = response_builder
        self.vault = vault
    
    async def list_campaigns(
        self,
        status: Optional[str] = None,
        level: Optional[int] = None
    ) -> Dict[str, Any]:
        """List campaigns with optional filters"""
        
        try:
            # Get campaigns from vault
            campaigns = await self.vault.list_campaigns(level=level, status=status)
            
            # Format campaign summaries
            campaign_list = []
            for campaign in campaigns:
                # Calculate summary metrics
                total_tests = len(campaign.results)
                successful_tests = sum(1 for r in campaign.results if r.get("success", False))
                
                summary = {
                    "id": campaign.id,
                    "name": campaign.name,
                    "description": campaign.description,
                    "level": campaign.level,
                    "status": campaign.status,
                    "created_at": datetime.fromtimestamp(campaign.created_at).isoformat(),
                    "variants": len(campaign.variants),
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "success_rate": round(successful_tests / total_tests, 2) if total_tests > 0 else 0
                }
                
                # Add level-specific info
                if campaign.level == 2:
                    summary["unlocked"] = campaign.metadata.get("unlocked", False)
                elif campaign.level == 3:
                    summary["revealed"] = campaign.metadata.get("revealed", False)
                    summary["hypothesis"] = campaign.metadata.get("hypothesis", "")
                
                campaign_list.append(summary)
            
            # Group by status
            status_groups = {}
            for c in campaign_list:
                status = c["status"]
                if status not in status_groups:
                    status_groups[status] = []
                status_groups[status].append(c)
            
            # Build response
            response_data = {
                "total_campaigns": len(campaign_list),
                "campaigns": campaign_list,
                "by_status": status_groups,
                "filters_applied": {
                    "status": status,
                    "level": level
                }
            }
            
            # Add suggestions based on results
            suggestions = []
            
            # Suggest running created campaigns
            created_campaigns = status_groups.get("created", [])
            if created_campaigns:
                latest_created = created_campaigns[0]  # Already sorted by creation time
                suggestions.append(
                    self.response_builder.suggest_next(
                        "akab_execute_campaign",
                        f"Execute '{latest_created['name']}'",
                        campaign_id=latest_created["id"],
                        iterations=5
                    )
                )
            
            # Suggest analyzing completed campaigns
            completed_campaigns = status_groups.get("completed", [])
            if completed_campaigns:
                for c in completed_campaigns[:2]:  # Top 2
                    suggestions.append(
                        self.response_builder.suggest_next(
                            "akab_analyze_results",
                            f"Analyze '{c['name']}'",
                            campaign_id=c["id"]
                        )
                    )
            
            # Suggest creating new campaign if none exist
            if not campaign_list:
                suggestions.append(
                    self.response_builder.suggest_next(
                        "akab_create_campaign",
                        "Create your first campaign",
                        name="My First A/B Test",
                        description="Testing different approaches"
                    )
                )
            
            return self.response_builder.success(
                data=response_data,
                message=f"Found {len(campaign_list)} campaigns",
                suggestions=suggestions
            )
            
        except Exception as e:
            return self.response_builder.error(str(e))
