import neuralcoref
import pandas as pd
import spacy

from src.graphdb_base import GraphDBBase
from src.text_processing_knowledge_graph import TextProcessor


class GraphBasedNLP(GraphDBBase):

    def __init__(self, database):
        super().__init__(database=database)
        self.nlp = spacy.load("en_core_web_sm")
        coref = neuralcoref.NeuralCoref(self.nlp.vocab)
        self.nlp.add_pipe(coref, name="neuralcoref")
        self.__text_processor = TextProcessor(self.nlp, self.get_session)

    def create_constraints(self):
        self.execute_without_exception("CREATE CONSTRAINT ON (u:Tag) ASSERT (u.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (i:TagOccurrence) ASSERT (i.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (t:Sentence) ASSERT (t.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:AnnotatedText) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:NamedEntity) ASSERT (l.id) IS NODE KEY")

    def import_masc(self, file):
        j = 0
        for chunk in pd.read_csv(file, header=None, sep="\t", chunksize=10 ** 3):
            df = chunk
            for record in df.to_dict("records"):
                record.copy()
                j += 1
                # self.tokenize_and_store(row[6],j,False)
                if j % 1000 == 0:
                    print(j, "lines processed")

        print(j, "total lines")

    def tokenize_and_store(self, text, text_id, store_tag):
        docs = self.nlp.pipe([text])
        for doc in docs:
            annotated_text = self.__text_processor.create_annotated_text(doc, text_id)
            spans = self.__text_processor.process_sentences(annotated_text, doc, store_tag, text_id)
            self.__text_processor.process_entities(spans, text_id)
            self.__text_processor.process_coreference(doc, text_id)

    def create_constraints_with_inferred_knowledge(self):
        self.execute_without_exception("CREATE CONSTRAINT ON (u:Tag) ASSERT (u.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (i:TagOccurrence) ASSERT (i.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (t:Sentence) ASSERT (t.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:AnnotatedText) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:NamedEntity) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:Entity) ASSERT (l.type, l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:Evidence) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:Relationship) ASSERT (l.id) IS NODE KEY")

    def tokenize_and_store_with_inferred_knowledge(self, text, text_id, store_tag):
        docs = self.nlp.pipe([text])
        for doc in docs:
            annotated_text = self.__text_processor.create_annotated_text(doc, text_id)
            spans = self.__text_processor.process_sentences(annotated_text, doc, store_tag, text_id)
            self.__text_processor.process_entities(spans, text_id)
            self.__text_processor.process_coreference(doc, text_id)
            self.__text_processor.build_entities_inferred_graph(text_id)
            rules = [
                {
                    "type": "RECEIVE_PRIZE",
                    "verbs": ["receive"],
                    "subjectTypes": ["PERSON", "NP"],
                    "objectTypes": ["WORK_OF_ART"],
                },
            ]
            self.__text_processor.extract_relationships(text_id, rules)
            self.__text_processor.build_relationships_inferred_graph(text_id)


def run_ingest_tag_with_inferred_knowledge():
    basic_nlp = GraphBasedNLP("spacyner")
    store_tag = False
    basic_nlp.create_constraints_with_inferred_knowledge()
    sentences = [
        "Marie Curie received the Nobel Prize in Physics in 1903. " +
        "She became the first woman to win the prize and the first person—man or woman—to win the award twice.",
        "President Barack Obama was born in Hawaii.  He was elected president in 2008.",
    ]
    for idx, x in enumerate(sentences):
        basic_nlp.tokenize_and_store_with_inferred_knowledge(x, idx, store_tag)

    basic_nlp.close()


if __name__ == "__main__":
    run_ingest_tag_with_inferred_knowledge()
