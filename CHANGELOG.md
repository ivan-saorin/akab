# Changelog

## [2.0.0] - 2025-01-18

### Architecture Documentation
- Updated ASR (Architecture Significant Records) to v2.0
- Added ASR-013: Substrate Foundation Architecture
- Added ASR-014: Evaluation Engine Architecture
- Added ASR-015: FileSystem Architecture Evolution
- Updated existing ASRs to reflect Substrate integration

### Changed - MAJOR ARCHITECTURE REFACTOR
- **BREAKING**: Refactored to use Substrate as foundation layer
- **BREAKING**: Import paths changed:
  - `akab.providers` → `providers` (from substrate)
  - `akab.evaluation` → `evaluation` (from substrate)
  - `mcp.server.fastmcp.FastMCP` → `mcp.server.FastMCP`
- Dockerfile now uses `FROM substrate:latest` instead of building everything
- FileSystemManager split into base (substrate) + AKAB-specific extensions

### Added
- Substrate foundation layer providing:
  - MCP server implementation
  - Multi-provider support
  - Evaluation engine
  - Base filesystem utilities
- `AKABFileSystemManager` extending substrate's base functionality
- Build scripts for both Windows (`build-all.ps1`) and Linux (`build-all.sh`)
- Verification script (`verify_setup.py`)
- Migration guide (`MIGRATION.md`)
- Architecture examples (`EXAMPLE.md`)

### Improved
- Cleaner separation of concerns
- Reusable components for other projects
- Faster Docker builds through layer caching
- Better maintainability

### Fixed
- Removed code duplication between projects
- Consistent imports across all modules

## [1.0.0] - Previous Version
- Initial AKAB implementation
- All components bundled together
