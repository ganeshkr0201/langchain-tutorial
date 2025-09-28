import requests
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated

load_dotenv()

message_history = []


# step 1: tool creation
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency.
    """
    url = f"https://v6.exchangerate-api.com/v6/05d018047c4fbbcbe678b747/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()


@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate

@tool
def sum(a: int, b: int) -> int:
    """
    it takes two numbers as input and returns the sum of both the numbers
    """
    return a + b

# step 2: tool binding
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
llm_with_tool = llm.bind_tools([get_conversion_factor, convert, sum])


# step 3: taking input from user
query = HumanMessage("can you tell me what indian ruppes i will get from 10 us dollors and what is 8 + 4")
message_history.append(query)

# step 4: tool calling 
while(True):
    ai_response = llm_with_tool.invoke(message_history)
    message_history.append(ai_response)

    if(ai_response.tool_calls):
        print(ai_response.tool_calls)
        # step 4: tool execution
        for tool in ai_response.tool_calls:
            if(tool['name'] == "sum"):
                tool_response = sum.invoke(tool)

            if(tool['name'] == "get_conversion_factor"):
                tool_response = get_conversion_factor.invoke(tool)
                conversion_rate = json.loads(tool_response.content)['conversion_rate'] 
    
            if(tool['name'] == "convert"):
                tool['args']['conversion_rate'] = conversion_rate
                tool_response = convert.invoke(tool)

            message_history.append(tool_response)
    else:
        break
    
print(ai_response.content)