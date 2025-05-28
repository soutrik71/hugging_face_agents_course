import gradio as gr
import random
from smolagents import (
    GradioUI,
    CodeAgent,
    HfApiModel,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
)

# Import our custom tools from their modules
from tools import WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset
import os

from dotenv import load_dotenv

load_dotenv()
# Initialize the Hugging Face model
model = HfApiModel(token=os.getenv("HF_TOKEN"))

# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()
search_tool.description = "A tool to search the web for any topic or person but this tool is of lower priority than the guest info tool, so it will only be used if the guest info tool does not have relevant information."

# Initialize the weather tool
weather_info_tool = WeatherInfoTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the guest dataset and initialize the guest info tool
guest_info_tool = load_guest_dataset()

# Create Alfred with all the tools
alfred = ToolCallingAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=True,
    planning_interval=1,
    max_steps=7,
    name="Alfred",
    description="A helpful assistant for the gala event, providing information about guests and thier relations, weather updates, and model stats in Hugging Face Hub.",
)

if __name__ == "__main__":
    GradioUI(alfred).launch()
