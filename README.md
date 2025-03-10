# RAG

```
from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.inputs import StrInput, SecretStrInput, MultilineInput, IntInput
from langflow.schema import Data

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
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("Could not import neo4j package. Please install it with `pip install neo4j`.")
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        return driver

    def search_documents(self) -> list[Data]:
        driver = self.build_vector_store()
        if self.search_input and isinstance(self.search_input, str) and self.search_input.strip():
            try:
                with driver.session() as session:
                    result = session.run(self.search_input)
                    docs = []
                    for record in result:
                        keys = list(record.keys())
                        # Prefer the key 'n' if it exists; otherwise, use the first key.
                        if "n" in keys:
                            docs.append(record["n"])
                        elif keys:
                            docs.append(record[keys[0]])
            except Exception as e:
                raise ValueError(f"Error performing search in Neo4j: {str(e)}") from e

            data = docs_to_data(docs)
            self.status = data
            return data
        else:
            return []
```
