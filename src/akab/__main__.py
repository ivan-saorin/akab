"""Entry point for akab MCP server"""

import sys
from .server import AkabServer


def main():
    """Main entry point"""
    try:
        server = AkabServer()
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
