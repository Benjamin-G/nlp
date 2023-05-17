import configparser
import os
from contextlib import suppress

from neo4j import GraphDatabase


class GraphDBBase:
    def __init__(self, database: str = "spacy"):
        self.database = database
        
        config = configparser.ConfigParser()
        config_file = os.path.join(os.path.dirname(__file__), "..", "config.ini")
        config.read(config_file)

        uri = "bolt://localhost:7687"
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._session = None

    def close(self):
        self._driver.close()

    def get_session(self):
        return self._driver.session(database=self.database)

    def execute_without_exception(self, query: str):
        with suppress(Exception):
            self.get_session().run(query)

    def execute_no_exception(self, session, query: str):
        with suppress(Exception):
            session.run(query)
