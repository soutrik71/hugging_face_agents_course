from smolagents import DuckDuckGoSearchTool
from smolagents import Tool
import random
from huggingface_hub import list_models


# Initialize the DuckDuckGo search tool
# search_tool = DuckDuckGoSearchTool()


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = (
        "Fetches dummy weather information for a given location during the gala event."
    )
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for.",
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # Dummy weather data
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20},
        ]
        # Randomly select a weather condition
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from.",
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # List models from the specified author, sorted by downloads
            models = list(
                list_models(author=author, sort="downloads", direction=-1, limit=1)
            )

            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"


if __name__ == "__main__":
    import os
    from smolagents import HfApiModel, ToolCallingAgent
    from dotenv import load_dotenv

    load_dotenv()

    tool_model = HfApiModel(
        model_id="Qwen/Qwen2.5-7B-Instruct", token=os.getenv("HF_TOKEN")
    )

    # Initialize the agent with the tools
    agent = ToolCallingAgent(
        model=tool_model,
        tools=[
            WeatherInfoTool(),
            HubStatsTool(),
            DuckDuckGoSearchTool(),
        ],
        name="GalaAssistant",
        description="An assistant for the gala event, providing weather updates and model stats in Hugging Face Hub.",
        max_steps=7,
        planning_interval=1,
    )

    # Example query Alfred might receive during the gala
    response = agent.run(
        "What is Meta Llama and what's their most popular model in hugging face hub?"
    )

    print("🎩 agent's Response:")
    print("========================================")
    print(response)
    print("========================================")
