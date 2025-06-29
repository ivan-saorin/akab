"""Main entry point for AKAB MCP server."""

import logging
import os
import sys

from .server import AKABServer

if __name__ == "__main__":
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr  # Important for Docker/MCP
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting AKAB MCP Server v2.0.0")
    logger.info(f"Transport: {os.getenv('MCP_TRANSPORT', 'stdio')}")
    logger.info(f"Data path: {os.getenv('AKAB_DATA_PATH', './akab_data')}")
    
    # Check API keys
    api_keys_configured = {
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Google": bool(os.getenv("GOOGLE_API_KEY")),
    }
    
    for provider, configured in api_keys_configured.items():
        if configured:
            logger.info(f"{provider} API key: Configured")
        else:
            logger.warning(f"{provider} API key: Not configured")
    
    # Create server and let FastMCP handle the event loop
    server = AKABServer()
    
    try:
        # FastMCP.run() will create its own event loop
        server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("AKAB server stopped")
