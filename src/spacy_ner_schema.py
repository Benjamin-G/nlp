import spacy

from src.graphdb_base import GraphDBBase
from src.text_processing_knowledge_graph import TextProcessor


class GraphBasedNLP(GraphDBBase):

    def __init__(self):
        super().__init__(database="spacyner")
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        self.__text_processor = TextProcessor(self.nlp, self.get_session)
        self.create_constraints()

    def create_constraints(self):
        self.execute_without_exception("CREATE CONSTRAINT ON (u:Tag) ASSERT (u.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (i:TagOccurrence) ASSERT (i.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (t:Sentence) ASSERT (t.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:AnnotatedText) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:NamedEntity) ASSERT (l.id) IS NODE KEY")

    def tokenize_and_store(self, text, text_id, storeTag):
        docs = self.nlp.pipe([text])
        for doc in docs:
            annotated_text = self.__text_processor.create_annotated_text(doc, text_id)
            spans = self.__text_processor.process_sentences(annotated_text, doc, storeTag, text_id)
            self.__text_processor.process_entities(spans, text_id)


if __name__ == "__main__":
    basic_nlp = GraphBasedNLP()
    basic_nlp.tokenize_and_store("President Barack Obama was born in Hawaii.  He was elected president in 2008.", 1,
                                 False)
    basic_nlp.close()
