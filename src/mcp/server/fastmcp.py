"""
Minimal MCP (Model Context Protocol) implementation for AKAB
This provides a simplified FastMCP server for development
"""

from typing import Dict, Any, Callable, Optional, List
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import json
import logging
import asyncio
from functools import wraps
import uuid

logger = logging.getLogger(__name__)


class MCPTool:
    """Represents an MCP tool"""
    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description or func.__doc__ or ""
        self.is_async = asyncio.iscoroutinefunction(func)


class MCPResource:
    """Represents an MCP resource"""
    def __init__(self, uri: str, func: Callable):
        self.uri = uri
        self.func = func
        self.is_async = asyncio.iscoroutinefunction(func)


class FastMCP:
    """Simplified FastMCP implementation for AKAB"""
    
    def __init__(self, name: str = "MCP Server"):
        self.name = name
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self._app: Optional[FastAPI] = None
        self.server_info = {
            "name": name,
            "version": "1.0.0"
        }
    
    def tool(self, name: Optional[str] = None):
        """Decorator for registering MCP tools"""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            self.tools[tool_name] = MCPTool(tool_name, func)
            logger.info(f"Registered tool: {tool_name}")
            return func
        return decorator
    
    def resource(self, uri: str):
        """Decorator for registering MCP resources"""
        def decorator(func: Callable):
            self.resources[uri] = MCPResource(uri, func)
            logger.info(f"Registered resource: {uri}")
            return func
        return decorator
    
    def streamable_http_app(self) -> FastAPI:
        """Create a FastAPI app for HTTP streaming"""
        if self._app is None:
            self._app = self._create_app()
        return self._app
    
    async def _handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        try:
            if method == "initialize":
                # Handle initialization
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {"listChanged": True},
                            "resources": {"subscribe": False, "listChanged": True}
                        },
                        "serverInfo": self.server_info
                    }
                }
            
            elif method == "tools/list":
                # List available tools
                tools_list = []
                for name, tool in self.tools.items():
                    tools_list.append({
                        "name": name,
                        "description": tool.description,
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    })
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": tools_list
                    }
                }
            
            elif method == "tools/call":
                # Execute a tool
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                if tool_name not in self.tools:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Tool '{tool_name}' not found"
                        }
                    }
                
                tool = self.tools[tool_name]
                
                # Execute tool
                if tool.is_async:
                    result = await tool.func(**tool_args)
                else:
                    result = tool.func(**tool_args)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result) if not isinstance(result, str) else result
                            }
                        ]
                    }
                }
            
            elif method == "resources/list":
                # List available resources
                resources_list = []
                for uri, resource in self.resources.items():
                    resources_list.append({
                        "uri": uri,
                        "name": uri,
                        "mimeType": "application/json"
                    })
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "resources": resources_list
                    }
                }
            
            elif method == "resources/read":
                # Read a resource
                uri = params.get("uri")
                
                if uri not in self.resources:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": f"Resource '{uri}' not found"
                        }
                    }
                
                resource = self.resources[uri]
                
                # Execute resource function
                if resource.is_async:
                    result = await resource.func()
                else:
                    result = resource.func()
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(result) if not isinstance(result, str) else result
                            }
                        ]
                    }
                }
            
            else:
                # Method not found
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not found"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error handling MCP request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    def _create_app(self) -> FastAPI:
        """Create the FastAPI application"""
        app = FastAPI(title=self.name)
        
        @app.get("/")
        async def root():
            return {
                "name": self.name,
                "version": "1.0.0",
                "protocol": "mcp",
                "tools": list(self.tools.keys()),
                "resources": list(self.resources.keys())
            }
        
        @app.post("/mcp")
        async def mcp_endpoint(request: Request):
            """Main MCP protocol endpoint for streamable HTTP"""
            try:
                # Get the request body
                body = await request.body()
                
                # Handle streaming requests
                if request.headers.get("content-type") == "application/json":
                    # Single request
                    mcp_request = json.loads(body)
                    response = await self._handle_mcp_request(mcp_request)
                    return JSONResponse(content=response)
                else:
                    # Streaming request (line-delimited JSON)
                    async def generate():
                        for line in body.decode().strip().split('\n'):
                            if line:
                                mcp_request = json.loads(line)
                                response = await self._handle_mcp_request(mcp_request)
                                yield json.dumps(response) + '\n'
                    
                    return StreamingResponse(
                        generate(),
                        media_type="application/json-lines"
                    )
                    
            except Exception as e:
                logger.error(f"Error in MCP endpoint: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    }
                )
        
        @app.post("/mcp/tools/{tool_name}")
        async def execute_tool(tool_name: str, request: Request):
            """Execute an MCP tool (legacy endpoint)"""
            if tool_name not in self.tools:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Tool '{tool_name}' not found"}
                )
            
            tool = self.tools[tool_name]
            
            try:
                # Parse request body
                body = await request.json()
                params = body.get("params", {})
                
                # Execute tool
                if tool.is_async:
                    result = await tool.func(**params)
                else:
                    result = tool.func(**params)
                
                return JSONResponse(content={
                    "status": "success",
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": str(e)
                    }
                )
        
        @app.get("/mcp/resources/{uri:path}")
        async def get_resource(uri: str):
            """Get an MCP resource (legacy endpoint)"""
            if uri not in self.resources:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Resource '{uri}' not found"}
                )
            
            resource = self.resources[uri]
            
            try:
                if resource.is_async:
                    result = await resource.func()
                else:
                    result = resource.func()
                
                return JSONResponse(content=result)
                
            except Exception as e:
                logger.error(f"Error getting resource {uri}: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
        
        @app.get("/health")
        async def health():
            """Health check endpoint"""
            # Try to call the health resource if it exists
            if "health" in self.resources:
                resource = self.resources["health"]
                if resource.is_async:
                    return await resource.func()
                else:
                    return resource.func()
            
            # Default health response
            return {
                "status": "healthy",
                "server": self.name,
                "tools_count": len(self.tools),
                "resources_count": len(self.resources)
            }
        
        @app.get("/mcp/tools")
        async def list_tools():
            """List all available tools"""
            tools_info = []
            for name, tool in self.tools.items():
                tools_info.append({
                    "name": name,
                    "description": tool.description
                })
            return {"tools": tools_info}
        
        @app.get("/mcp/resources")
        async def list_resources():
            """List all available resources"""
            return {"resources": list(self.resources.keys())}
        
        return app
