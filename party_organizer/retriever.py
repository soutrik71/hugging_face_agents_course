from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
import datasets
from dotenv import load_dotenv

load_dotenv()

import os


hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Please set the HF_TOKEN environment variable in your .env file.")


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name, relation, description and email id .Useful for providing context about guests during the event."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        """
        Initializes the GuestInfoRetrieverTool with a list of Document objects.
        :param docs: List of Document objects containing guest information.
        """
        super().__init__(**kwargs)
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        """
        Retrieves relevant guest information based on the provided query.
        :param query: The name or relation of the guest to search for.
        :return: A string containing the relevant guest information.
        """
        if not query:
            return "Please provide a guest name or relation to search for."

        # Use the retriever to get relevant documents based on the query
        results = self.retriever.get_relevant_documents(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."


def load_guest_dataset():
    # Load the dataset
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

    # Convert dataset entries into Document objects
    docs = [
        Document(
            page_content="\n".join(
                [
                    f"Name: {guest['name']}",
                    f"Relation: {guest['relation']}",
                    f"Description: {guest['description']}",
                    f"Email: {guest['email']}",
                ]
            ),
            metadata={"name": guest["name"]},
        )
        for guest in guest_dataset
    ]
    print(f"Loaded {len(docs)} guest documents.")

    # Return the tool
    return GuestInfoRetrieverTool(docs)


if __name__ == "__main__":
    from smolagents import CodeAgent, HfApiModel, ToolCallingAgent, MultiStepAgent

    model = HfApiModel(token=os.getenv("HF_TOKEN"))

    agent = CodeAgent(
        tools=[load_guest_dataset()],
        model=model,
    )

    # Example query Alfred might receive during the gala
    response = agent.run(
        "Tell me about our guest named 'Lady Ada Lovelace' only from the context?",
    )

    print("🎩 agent's Response:")
    print("========================================")
    print(response)
    print("========================================")
