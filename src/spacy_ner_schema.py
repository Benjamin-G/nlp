import neuralcoref
import spacy

from src.graphdb_base import GraphDBBase
from src.text_processing_knowledge_graph import TextProcessor


class GraphBasedNLP(GraphDBBase):

    def __init__(self):
        super().__init__(database="spacyner")
        self.nlp = spacy.load("en_core_web_sm")
        coref = neuralcoref.NeuralCoref(self.nlp.vocab)
        self.nlp.add_pipe(coref, name="neuralcoref")
        self.__text_processor = TextProcessor(self.nlp, self.get_session)
        self.create_constraints()

    def create_constraints(self):
        self.execute_without_exception("CREATE CONSTRAINT ON (u:Tag) ASSERT (u.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (i:TagOccurrence) ASSERT (i.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (t:Sentence) ASSERT (t.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:AnnotatedText) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:NamedEntity) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:Entity) ASSERT (l.type, l.id) IS NODE KEY")

    def tokenize_and_store(self, text, text_id, storeTag):
        docs = self.nlp.pipe([text])
        for doc in docs:
            annotated_text = self.__text_processor.create_annotated_text(doc, text_id)
            spans = self.__text_processor.process_sentences(annotated_text, doc, storeTag, text_id)
            self.__text_processor.process_entities(spans, text_id)
            self.__text_processor.process_coreference(doc, text_id)
            self.__text_processor.build_entities_inferred_graph(text_id)


if __name__ == "__main__":
    basic_nlp = GraphBasedNLP()
    store_tag = False
    sentences = [
        "Marie Curie received the Nobel Prize in Physics in 1903. " +
        "She became the first woman to win the prize and the first person—man or woman—to win the award twice.",
        "President Barack Obama was born in Hawaii.  He was elected president in 2008.",
    ]
    for idx, x in enumerate(sentences):
        basic_nlp.tokenize_and_store(x, idx, store_tag)

    basic_nlp.close()
