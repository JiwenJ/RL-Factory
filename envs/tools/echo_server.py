from fastmcp import FastMCP

mcp = FastMCP(name="SimpleServer")

@mcp.tool()
def greet(name: str) -> str:
    return f"你好，{name}！"

@mcp.tool()
def hello(name: str) -> str:
    return f"你好，{name}！"

if __name__ == "__main__":
    # 指定端口为8080，host为0.0.0.0，传输协议为http
    mcp.run(transport="stdio")