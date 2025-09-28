from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


load_dotenv()

# message history
message_history = []



# step-1: tool creation
@tool
def multiply(a: int, b: int) -> int:
    """
        Takes two numbers as input and returns their products.
    """
    return a * b


# step-2: tool binding
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  
llm_with_tools = llm.bind_tools([multiply])


# step-3: tool calling
query = "can you multiply 4 and 8"
message_history.append(HumanMessage(query))

result = llm_with_tools.invoke(message_history)
message_history.append(result)


# step-4: tool execution
tool_result = multiply.invoke(result.tool_calls[0])
message_history.append(tool_result)


# step-5: llm call
final_result = llm_with_tools.invoke(message_history)
print(final_result.content)
