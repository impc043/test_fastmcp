from fastmcp  import FastMCP
import random

mcp = FastMCP("demo serveer")

@mcp.tool
def my_greet(name: str):
    """
    Docstring for my_greet
    
    :param name: Description
    :type name: str
    """

    return f"Hello {name}!" 