from dotenv import load_dotenv
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

# step 1: create tools
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"http://api.weatherstack.com/current?access_key=ea883200aea1c044c9ac4e7f06136a2d&query={city}"
    response = requests.get(url)

    return response.json()



# step 2: create llm
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


# step 3: create prompt
prompt = hub.pull('hwchase17/react')


# step 4: create agent
agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=[search_tool, get_weather_data]
)


# step 5: create agent executer
agent_executer = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)


# step 6: invoke executer
result = agent_executer.invoke({"input": "what is the capital of bihar and its current weather?"})


print(result['output'])