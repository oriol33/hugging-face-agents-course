import os
from dotenv import load_dotenv

from smolagents import CodeAgent, DuckDuckGoSearchTool
from smolagents import InferenceClientModel
from smolagents.monitoring import LogLevel

from tools import wiki_search
from tools import multiply, add, subtract, divide, modulus, power


# --- Agent assembling ---
class GaiaAgentWrapper:
    """
    A wrapper class for initializing an AI agent that runs GAIA questions, based on CodeAgent from smolagents.
    """

    def __init__(self):
        print("GaiaAgentWrapper initialized.")

        # Instantiate the model
        load_dotenv()  # load environment variables from .env, only needed in local environment
        api_hf_token = os.getenv("HF_TOKEN")

        model_id = "meta-llama/Llama-3.3-70B-Instruct" # default: Qwen/Qwen2.5-Coder-32B-Instruct, others: meta-llama/Llama-3.3-70B-Instruct
        self.model = InferenceClientModel(model_id=model_id, token=api_hf_token)

        # Instantiate agent tools
        self.search_tool = DuckDuckGoSearchTool()
        self.wiki_search = wiki_search

        # Add GAIA suggested prompt instructions
        flag_gaia_instructions = True
        gaia_instructions = "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."

        if flag_gaia_instructions:
            instructions = gaia_instructions
        else: instructions = None

        # Declare additional imports for the agent
        add_imports = ["pandas", "markdownify", "requests"]

        # Create the agent
        self.agent = CodeAgent(
        tools=[
            self.search_tool,
            self.wiki_search,
            multiply,
            add,
            subtract,
            divide,
            modulus,
            power,
            ], 
        model=self.model,
        instructions=instructions,
        add_base_tools=True,                 # Add any additional base tools
        verbosity_level=LogLevel.INFO,       # default: LogLevel.INFO, LogLevel.ERROR to supress verbosity to minimum
        planning_interval=3,                 # Enable planning every 3 steps
        additional_authorized_imports=add_imports
    )

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        # Run the agent provided with the question
        response = self.agent.run(question)

        print(f"Agent returning response: {response}")
        return response