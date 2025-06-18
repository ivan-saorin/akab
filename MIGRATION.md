# AKAB Migration Guide: Substrate Architecture

## Overview

AKAB has been refactored to use Substrate as its foundational layer. This provides:
- Better modularity and reusability
- Cleaner separation of concerns
- Easier maintenance and testing
- Foundation for other AI projects (Thrice, Synapse, etc.)

## What Changed

### 1. Directory Structure

**Before:**
```
akab/
├── src/
│   ├── mcp/          # Duplicated
│   ├── akab/
│   │   ├── providers.py  # Duplicated
│   │   ├── evaluation.py # Duplicated
│   │   └── filesystem.py # Mixed concerns
```

**After:**
```
substrate/
├── src/
│   ├── mcp/          # Shared
│   ├── providers/    # Shared
│   ├── evaluation/   # Shared
│   └── filesystem/   # Base utilities

akab/
├── src/
│   └── akab/
│       ├── filesystem.py  # AKAB-specific, extends substrate
│       ├── server.py      # Imports from substrate
│       └── tools/         # AKAB-specific tools
```

### 2. Imports

**Before:**
```python
from mcp.server.fastmcp import FastMCP
from akab.providers import ProviderManager
from akab.evaluation import EvaluationEngine
```

**After:**
```python
from mcp.server import FastMCP        # From substrate
from providers import ProviderManager  # From substrate
from evaluation import EvaluationEngine # From substrate
from akab.filesystem import AKABFileSystemManager  # AKAB-specific
```

### 3. Docker Build Process

**Before:**
```bash
docker build -t akab:latest .
```

**After:**
```bash
# Build substrate first (one-time)
cd ../substrate
docker build -t substrate:latest .

# Then build AKAB
cd ../akab
docker build -t akab:latest .
```

### 4. FileSystem Manager

The FileSystemManager is now split:
- **Base FileSystemManager** (substrate): Generic file operations
- **AKABFileSystemManager** (akab): Campaign/experiment specific operations

## Benefits

1. **Reusability**: Substrate components can be used by other projects
2. **Maintenance**: Bug fixes in substrate benefit all projects
3. **Testing**: Components can be tested independently
4. **Docker Layers**: Shared base image reduces build times
5. **Clear Boundaries**: AKAB-specific vs generic utilities

## Migration Steps

1. **Pull latest code**
2. **Build substrate base image**: `cd substrate && docker build -t substrate:latest .`
3. **Build AKAB**: `cd akab && docker build -t akab:latest .`
4. **Update any custom scripts** to use new import paths
5. **Test with**: `python test_substrate.py`

## Future Projects Using Substrate

```
substrate:latest
├── akab:latest      # Experiment management
├── thrice:latest    # The phenomenon engine
├── synapse:latest   # AI interaction tool
└── your-project:latest
```

## Breaking Changes

1. Import paths have changed (see above)
2. `akab.providers` → `providers` (from substrate)
3. `akab.evaluation` → `evaluation` (from substrate)
4. FileSystemManager split into base + AKAB-specific

## Questions?

If you encounter issues:
1. Ensure substrate is built first
2. Check import paths match the new structure
3. Verify Docker base image is `FROM substrate:latest`
