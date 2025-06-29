#!/bin/bash
# Build script for AKAB Docker image

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="akab-mcp"
VERSION="${VERSION:-latest}"
REGISTRY="${REGISTRY:-}"

echo -e "${GREEN}Building AKAB MCP Docker Image${NC}"
echo "Version: $VERSION"

# Ensure we're in the AKAB directory
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found!${NC}"
    echo "Please run this script from the AKAB directory"
    exit 1
fi

# Build image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build \
    -f Dockerfile \
    -t ${IMAGE_NAME}:${VERSION} \
    -t ${IMAGE_NAME}:latest \
    .

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Tag for registry if specified
if [ -n "$REGISTRY" ]; then
    echo -e "${YELLOW}Tagging for registry: $REGISTRY${NC}"
    docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:latest
fi

# Show image info
echo -e "${GREEN}Build complete!${NC}"
docker images | grep ${IMAGE_NAME} || true

# Test the image
echo -e "${YELLOW}Testing image...${NC}"
docker run --rm ${IMAGE_NAME}:${VERSION} python -c "import akab; print('AKAB import successful')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Success! Image is ready.${NC}"
    echo ""
    echo "To run AKAB:"
    echo "  docker run --rm -i \\"
    echo "    -e ANTHROPIC_API_KEY=your_key \\"
    echo "    -e OPENAI_API_KEY=your_key \\"
    echo "    -e GOOGLE_API_KEY=your_key \\"
    echo "    ${IMAGE_NAME}:${VERSION}"
else
    echo -e "${RED}Test failed!${NC}"
    exit 1
fi
