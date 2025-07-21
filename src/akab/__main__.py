"""Entry point for AKAB FastMCP server"""

import sys
from .server_fastmcp import mcp, register_all_features


def main():
    """Main entry point for AKAB"""
    try:
        # Register all features
        register_all_features()
        
        # Run the FastMCP server
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        print("\nShutting down AKAB...", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"AKAB Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
