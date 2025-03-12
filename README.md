# RAG

# Wait 60 seconds before connecting using these details, or login to https://console.neo4j.io to validate the Aura Instance is available
NEO4J_URI=neo4j+s://dd99637a.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=NGQCYgUrapBcKVzpgv3gfK33DweH9lvEFOv95p-wECY
AURA_INSTANCEID=dd99637a
AURA_INSTANCENAME=Instance01

```
from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.inputs import StrInput, SecretStrInput, MultilineInput, IntInput
from langflow.schema import Data
from langchain_community.graphs.neo4j_graph import Neo4jGraph

# Dummy implementation for converting Neo4j nodes to LangFlow Data objects.
# Replace this with your actual conversion logic as needed.
def docs_to_data(docs):
    return [Data(value=str(doc)) for doc in docs]

class Neo4jVectorStoreComponent(LCVectorStoreComponent):
    display_name: str = "Neo4j"
    description: str = "Vector Store component using Neo4j for search and ingestion"
    documentation: str = "https://neo4j.com/docs/"
    name = "Neo4j"
    icon: str = "Neo4j"
    
    inputs = [
        StrInput(
            name="uri",
            display_name="Neo4j URI",
            info="The URI for your Neo4j instance.",
            required=True,
        ),
        SecretStrInput(
            name="password",
            display_name="Password",
            info="Your Neo4j password.",
            required=True,
        ),
        StrInput(
            name="username",
            display_name="Username",
            info="Your Neo4j username.",
            required=True,
        ),
        StrInput(
            name="database_name",
            display_name="Database Name",
            info="Name of the Neo4j database (e.g., 'neo4j').",
            required=True,
        ),
        MultilineInput(
            name="search_input",
            display_name="Search Input (CQL)",
            info="Enter the full CQL query to extract data from the graph. For example, 'MATCH (n:Document) RETURN n LIMIT 10'.",
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="This parameter is available for compatibility but isn’t used when supplying a full CQL query.",
            value=4,
        ),
    ]

    @check_cached_vector_store
    def build_vector_store(self):
        return Neo4jGraph(
            url=self.uri,
            username=self.username,
            password=self.password,
            database=self.database_name,
        )

    def search_documents(self) -> list[Data]:
        graph = self.build_vector_store()
        if self.search_input and isinstance(self.search_input, str) and self.search_input.strip():
            try:
                result = graph.query(self.search_input)
                docs = [record for record in result]
            except Exception as e:
                raise ValueError(f"Error performing search in Neo4j: {str(e)}") from e

            data = docs_to_data(docs)
            self.status = data
            return data
        else:
            return []

```
