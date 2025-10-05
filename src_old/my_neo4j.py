from typing import List

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from pydantic import BaseModel


class Neo4jServer(BaseModel):
    hostname: str
    port: int
    username: str
    password: str
    database: str


class Neo4jClient:
    def __init__(self, ref_neo4j_server: Neo4jServer):
        self.server = ref_neo4j_server
        self.client = None

    @property
    def dsn(self):
        return f"bolt://{self.server.hostname}:{self.server.port}"

    def _connect(self):
        if not self.client:
            try:
                self.client = GraphDatabase.driver(
                    self.dsn, auth=(self.server.username, self.server.password)
                )
            except ServiceUnavailable as e:
                print("Connection error:", e)
                raise

    @property
    def session(self):
        self._connect()
        return self.client.session(database=self.server.database)

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

    def insertMany(self, nodes: List[dict], node_label: str = "Node"):
        """
        Inserts multiple nodes into Neo4j.

        Args:
            nodes: A list of dictionaries, where each dictionary represents a node's properties.
            node_label: The label to assign to the nodes.
        """
        query = f"UNWIND $nodes AS node CREATE (n:{node_label}) SET n += node"
        self.session.run(query, nodes=nodes)

    def calculateCosineSimilarity(
        self, node_label: str, key_identifier: str, property_name: str
    ):
        """
        Calculates the cosine similarity between nodes based on a specified property.

        Args:
            node_label: The label of the nodes to compare.
            property_name: The property on each node to use for cosine similarity.
        Returns:
            List of dictionaries with similarity vectors.
        """
        query = f"""
        MATCH (n1:{node_label}), (n2:{node_label})
        WHERE n1.{key_identifier} < n2.{key_identifier}
            AND n1.{property_name} IS NOT NULL
            AND n2.{property_name} IS NOT NULL
        RETURN n1.{key_identifier} AS Node1, n2.{key_identifier} AS Node2,
           gds.similarity.cosine(n1.{property_name}, n2.{property_name}) AS similarity
        ORDER BY similarity DESC
        """
        result = self.session.run(query)
        return [record.data() for record in result]


def main():
    neo4j_server = Neo4jServer(
        hostname="localhost",
        port=7687,
        username="neo4j",
        password="your_password",
        database="neo4j",
    )
    with Neo4jClient(neo4j_server) as client:
        nodes = [
            {"node_id": 0, "vector": [1.0, 1.0, 1.0]},
            {"node_id": 1, "vector": [1.0, 1.0, 1.0]},
            {"node_id": 2, "vector": [1.0, 1.0, 1.0]},
            {"node_id": 3, "vector": [1.0, 1.0, 0.0]},
            {"node_id": 4, "vector": [0.0, 1.0, 0.0]},
            {"node_id": 5, "vector": [1.0, 0.0, 1.0]},
        ]
        client.insertMany(nodes, node_label="MyNode")
        similarities = client.calculateCosineSimilarity(
            node_label="MyNode", key_identifier="node_id", property_name="vector"
        )
        print("Cosine Similarity Results:", similarities)


if __name__ == "__main__":
    main()
