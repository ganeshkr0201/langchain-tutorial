from langchain_core.tools import tool


# custom tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@tool
def check_even(a: int) -> bool:
    """Checks if a number is even or not"""
    return a % 2 == 0


class MathToolkit:
    def get_tools(self):
        return [add, multiply, check_even]
    

toolkit = MathToolkit()

tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "=>", tool.description)