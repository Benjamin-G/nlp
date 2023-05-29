import pandas as pd
import pytextrank
import spacy

from src.graphdb_base import GraphDBBase
from src.text_processing_knowledge_graph import TextProcessor
from src.utils import get_data_path


class GraphBasedNLP(GraphDBBase):

    def __init__(self, database):
        super().__init__(database=database)
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        # coref = neuralcoref.NeuralCoref(self.nlp.vocab)
        # self.nlp.add_pipe(coref, name='neuralcoref');
        # tr = pytextrank.TextRank()
        print(pytextrank.__version__)
        self.nlp.add_pipe("textrank")
        self.__text_processor = TextProcessor(self.nlp, self.get_session)

    def create_constraints(self):
        self.execute_without_exception("CREATE CONSTRAINT ON (u:Tag) ASSERT (u.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (i:TagOccurrence) ASSERT (i.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (t:Sentence) ASSERT (t.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:AnnotatedText) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:NamedEntity) ASSERT (l.id) IS NODE KEY")
        self.execute_without_exception("CREATE CONSTRAINT ON (l:Keyword) ASSERT (l.id) IS NODE KEY")

    def import_data(self, file):
        j = 0
        for chunk in pd.read_csv(file,
                                 header=None,
                                 skiprows=1,
                                 chunksize=10 ** 3):
            df = chunk
            print("Processing chunk")
            print(df.info())
            for record in df.to_dict("records"):
                row = record.copy()
                j += 1
                self.tokenize_and_store(row[7], j, False)
                if j % 100 == 0:
                    print(j, "lines processed")

                if j % 500 == 0:
                    return

        print(j, "total lines")

    def tokenize_and_store(self, text, text_id, store_tag):
        docs = self.nlp.pipe([text])
        for doc in docs:
            annotated_text = self.__text_processor.create_annotated_text(doc, text_id)
            spans = self.__text_processor.process_sentences(annotated_text, doc, store_tag, text_id)
            self.__text_processor.process_entities(spans, text_id)
            # self.process_coreference(doc, text_id)
            self.__text_processor.process_textrank(doc, text_id)


if __name__ == "__main__":
    basic_nlp = GraphBasedNLP("textrank-spacy")
    basic_nlp.create_constraints()
    data = get_data_path("wiki_movie_plots_deduped.csv")
    basic_nlp.import_data(data)
    basic_nlp.close()
