import pandas as pd
import pynvml
import pytextrank
import spacy
from prefect import flow, task
from prefect_dask import DaskTaskRunner

from src.graphdb_base import GraphDBBase
from src.text_processing_knowledge_graph import TextProcessor
from src.utils import get_data_path


@task()
def tokenize_and_store(text, text_id, store_tag, nlp=None, text_processor=None):
    docs = nlp.pipe([text])
    for doc in docs:
        annotated_text = text_processor.create_annotated_text(doc, text_id)
        spans = text_processor.process_sentences(annotated_text, doc, store_tag, text_id)
        text_processor.process_entities(spans, text_id)
        #  process_coreference(doc, text_id)
        text_processor.process_textrank(doc, text_id)


def create_constraints(graph_db: GraphDBBase):
    graph_db.execute_without_exception("CREATE CONSTRAINT ON (u:Tag) ASSERT (u.id) IS NODE KEY")
    graph_db.execute_without_exception("CREATE CONSTRAINT ON (i:TagOccurrence) ASSERT (i.id) IS NODE KEY")
    graph_db.execute_without_exception("CREATE CONSTRAINT ON (t:Sentence) ASSERT (t.id) IS NODE KEY")
    graph_db.execute_without_exception("CREATE CONSTRAINT ON (l:AnnotatedText) ASSERT (l.id) IS NODE KEY")
    graph_db.execute_without_exception("CREATE CONSTRAINT ON (l:NamedEntity) ASSERT (l.id) IS NODE KEY")
    graph_db.execute_without_exception("CREATE CONSTRAINT ON (l:Keyword) ASSERT (l.id) IS NODE KEY")


@flow(task_runner=DaskTaskRunner(), validate_parameters=False)
def run_nlp_pipeline_dask_task_runner():
    spacy.prefer_gpu()
    print("Running NLP pipeline")
    graph_db = GraphDBBase("textrank-spacy-prefect")
    create_constraints(graph_db)

    # Load the spacy model
    print(pytextrank.__version__)
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    # Create the text processor
    text_processor = TextProcessor(nlp, graph_db.get_session)

    # Run the pipeline
    file = get_data_path("wiki_movie_plots_deduped.csv")
    j = 0
    for chunk in pd.read_csv(file,
                             header=None,
                             skiprows=1,
                             chunksize=10 ** 3):
        print("Processing chunk")
        print(chunk.info())
        for text_line in chunk[7]:
            j += 1
            tokenize_and_store.submit(text_line, j, False, nlp, text_processor)
            # return
            if j % 100 == 0:
                print(j, "lines processed")

            if j % 500 == 0:
                return

        # df = chunk
        # print("Processing chunk")
        # print(df.info())
        # for record in df.to_dict("records"):
        #     row = record.copy()
        #     j += 1
        #     tokenize_and_store.submit(row[7], j, False, nlp, text_processor)
        #     if j % 100 == 0:
        #         print(j, "lines processed")
        # 
        #     if j % 500 == 0:
        #         return

    print(j, "total lines")


def run():
    # Create the graph database
    graph_db = GraphDBBase("textrank-spacy-prefect")
    create_constraints(graph_db)

    # Load the spacy model
    print(pytextrank.__version__)
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    # Create the text processor
    text_processor = TextProcessor(nlp, graph_db.get_session)

    # Run the pipeline
    data = get_data_path("wiki_movie_plots_deduped.csv")
    run_nlp_pipeline_dask_task_runner(data, nlp, text_processor)


if __name__ == "__main__":
    pynvml.nvmlInit()
    print(pynvml.nvmlDeviceGetCount())
    run_nlp_pipeline_dask_task_runner()
