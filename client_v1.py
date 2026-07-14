import asyncio
import json 
from langchain_mcp_adapters.client import MultiServerMCPClient
#as we want to connect to multiple servers
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import  load_dotenv
from langchain_core.messages import ToolMessage
load_dotenv()

#configure the servers you want to connect
SERVERS = { 
    "expenses": {
        "transport": "stdio",
        "command": "C:/Users/Lenovo/.local/bin/uv.exe",
        "args": [
            "run",
            "fastmcp",
            "run",
            "C:/Users/Lenovo/OneDrive/Desktop/MCP/expense-tracker-mcp-server/main.py"
        ]
    },
    "manim-server": {
        "transport": "stdio",
        "command": "C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe",
        "args": [
        "/Users/Lenovo/OneDrive/Desktop/MCP/manim-mcp-server/src/manim_server.py"
        ],
        "env": {
        "MANIM_EXECUTABLE": "/Library/Frameworks/Python.framework/Versions/3.11/bin/manim"
        }
    }
}

#async main func
async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    named_tools = {}
    #get tool list:
    for tool in tools:
        named_tools[tool.name] = tool
    #llm
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
    llm_with_tools = llm.bind_tools(tools)
    prompt = "What is product of 14 and 100"
    response = await llm_with_tools.ainvoke(prompt)
    
    #if tool calls is not required the llm gives ans else uses tools
    if not getattr(response,"tool_calls",None):
        print("\nLLM Reply :",response.content)
        return
    tool_messages = []
    #if llm has to use more tools 
    for tc in response.tool_calls:
        #fetch name args and id
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]
        #print(f"Selected tool:{selected_tool}")
        #print(f"Selected tool arguments : {selected_tool_args}")
    
        #to invoke the tools 
        #tool_result = await named_tools[selected_tool].ainvoke(**selected_tool_args)
        #print(f"Tool Result :{tool_result}")
        #sending back the llm the result using tool message 
        result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        #tool_message = ToolMessage(content=tool_result,tool_call_id = selected_tool_id)
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result)))
    #llm generates final response
    final_response = await llm_with_tools.ainvoke([prompt,response,*tool_messages])
    print(f"Final Response :{final_response.content}")
    
if __name__ == "main":
    asyncio.run(main())