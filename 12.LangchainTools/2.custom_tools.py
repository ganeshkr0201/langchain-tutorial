from langchain_core.tools import tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from typing import Type


# type 1: using @tool decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

result = multiply.invoke({'a': 3, 'b': 5})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())



# type 2: using StructuredTool & Pydantic
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a': 5, 'b': 6})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())



# type 3: using BaseTool class
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput
    def _run(self, a: int, b: int) -> int:
        return a * b
    
multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a': 3, 'b': 6})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())