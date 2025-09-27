from langchain_community.tools import DuckDuckGoSearchRun, ShellTool


# DuckDuckGo Web Search Tool
search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke("who is current prime minister of india?")
print(result)


# Shell Tool
shell_tool = ShellTool()

result = shell_tool.invoke("ls")
print(result)
