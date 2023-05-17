import spacy

from src.graphdb_base import GraphDBBase


class GraphBasedNLP(GraphDBBase):

    def __init__(self):
        super().__init__(database="spacy")
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        self.create_constraints()

    def create_constraints(self):
        with self._driver.session():
            self.execute_without_exception("CREATE CONSTRAINT ON (u:Tag) ASSERT (u.id) IS NODE KEY")
            self.execute_without_exception("CREATE CONSTRAINT ON (i:TagOccurrence) ASSERT (i.id) IS NODE KEY")
            self.execute_without_exception("CREATE CONSTRAINT ON (t:Sentence) ASSERT (t.id) IS NODE KEY")
            self.execute_without_exception("CREATE CONSTRAINT ON (l:AnnotatedText) ASSERT (l.id) IS NODE KEY")

    def tokenize_and_store(self, text, text_id, store_tag):
        """
        the resulting graph contains sentences, tokens—lemmatized, marked as stop words, and
        with PoS information—and relationships between tokens that describe their role in the sentence.
        
        * Next-word suggestion—As in section 11.1 with the next schema model, it is possible to suggest the next word 
        considering the current one or any number of previous words.
        
        * Advanced search engines—When we have the 
        information about the order of the words together with dependencies among them, we can implement advanced 
        search capabilities in which, apart from checking for the exact order of the words, it is possible to 
        consider cases with some words between our target and provide some suggestion. A concrete example follows 
        this list. 
        
        * Content-based recommendation—By decomposing the text into components, we can compare item 
        descriptions (movies, products, and so on). This step is one of the first required for providing 
        content-based recommendations. In this case, having the lemmatization and other normalization in place (
        stop-word removal, punctuation handling, and so on) will make the comparisons even more accurate. 
        :param text: 
        :type text: 
        :param text_id: 
        :type text_id: 
        :param store_tag: 
        :type store_tag: 
        :return: 
        :rtype:
        """
        docs = self.nlp.pipe([text], disable=["ner"])
        for doc in docs:
            annotated_text = self.create_annotated_text(doc, text_id)
            i = 1
            for sentence in doc.sents:
                print("-------- Sentence ", i, "-----------")
                print(annotated_text)
                print(sentence.text)
                self.store_sentence(sentence, annotated_text, text_id, i, store_tag)
                i += 1

    def create_annotated_text(self, doc, id):
        query = """MERGE (ann:AnnotatedText {id: $id})
            RETURN id(ann) as result
        """
        params = {"id": id}
        results = self.execute_query(query, params)
        return results[0]

    def store_sentence(self, sentence, annotated_text, text_id, sentence_id, store_tag):
        sentence_query = """
        MATCH (ann:AnnotatedText) WHERE id(ann) = $ann_id
            MERGE (sentence:Sentence {id: $sentence_unique_id})
        SET sentence.text = $text
            MERGE (ann)-[:CONTAINS_SENTENCE]->(sentence)
        RETURN id(sentence) as result
        """

        params = {
            "ann_id": annotated_text,
            "text": sentence.text,
            "sentence_unique_id": str(text_id) + "_" + str(sentence_id),
        }

        results = self.execute_query(sentence_query, params)
        node_sentence_id = results[0]

        tag_occurrence_query = """
        MATCH (sentence:Sentence) WHERE id(sentence) = $sentence_id
        WITH sentence, $tag_occurrences as tags
        FOREACH ( idx IN range(0,size(tags)-2) |
            MERGE (tagOccurrence1:TagOccurrence {id: tags[idx].id})
        SET tagOccurrence1 = tags[idx]
            MERGE (sentence)-[:HAS_TOKEN]->(tagOccurrence1)
            MERGE (tagOccurrence2:TagOccurrence {id: tags[idx + 1].id})
        SET tagOccurrence2 = tags[idx + 1]
            MERGE (sentence)-[:HAS_TOKEN]->(tagOccurrence2)
            MERGE (tagOccurrence1)-[r:HAS_NEXT {sentence: sentence.id}]->(tagOccurrence2))
        RETURN id(sentence) as result
        """

        tag_occurrence_with_tag_query = """
        MATCH (sentence:Sentence) WHERE id(sentence) = $sentence_id
        WITH sentence, $tag_occurrences as tags
        FOREACH ( idx IN range(0,size(tags)-2) |
            MERGE (tagOccurrence1:TagOccurrence {id: tags[idx].id})
        SET tagOccurrence1 = tags[idx]
            MERGE (sentence)-[:HAS_TOKEN]->(tagOccurrence1)
            MERGE (tagOccurrence2:TagOccurrence {id: tags[idx + 1].id})
        SET tagOccurrence2 = tags[idx + 1]
            MERGE (sentence)-[:HAS_TOKEN]->(tagOccurrence2)
            MERGE (tagOccurrence1)-[r:HAS_NEXT {sentence: sentence.id}]->(tagOccurrence2))
        FOREACH (tagItem in [tag_occurrence IN $tag_occurrences WHERE tag_occurrence.is_stop = False] | 
            MERGE (tag:Tag {id: tagItem.lemma}) 
            MERGE (tagOccurrence:TagOccurrence {id: tagItem.id}) 
            MERGE (tag)<-[:REFERS_TO]-(tagOccurrence))
        RETURN id(sentence) as result
        """

        tag_occurrences = []
        tag_occurrence_dependencies = []

        """the root (the main verb) of the dependency tree is recognizable via the self loop the new relations 
        connect TagOccurrence nodes to the dependent nodes. This connection is necessary because the same Tag can 
        have different relationships in different sentences (John might be the subject in some sentences and the 
        object in others), whereas a TagOccurrence represents the tag in a specific sentence context and can have 
        only a specific role."""

        for token in sentence:
            lexeme = self.nlp.vocab[token.text]
            if not lexeme.is_punct and not lexeme.is_space:
                tag_occurrence_id = str(text_id) + "_" + str(sentence_id) + "_" + str(token.idx)
                tag_occurrence = {"id": tag_occurrence_id,
                                  "index": token.idx,
                                  "text": token.text,
                                  "lemma": token.lemma_,
                                  "pos": token.tag_,
                                  "is_stop": (lexeme.is_stop or lexeme.is_punct or lexeme.is_space)}
                tag_occurrences.append(tag_occurrence)
                tag_occurrence_dependency_source = str(text_id) + "_" + str(sentence_id) + "_" + str(token.head.idx)
                dependency = {"source": tag_occurrence_dependency_source, "destination": tag_occurrence_id,
                              "type": token.dep_}
                tag_occurrence_dependencies.append(dependency)

        params = {"sentence_id": node_sentence_id, "tag_occurrences": tag_occurrences}
        if store_tag:
            results = self.execute_query(tag_occurrence_with_tag_query, params)
        else:
            results = self.execute_query(tag_occurrence_query, params)

        self.process_dependencies(tag_occurrence_dependencies)
        return results[0]

    def process_dependencies(self, tag_occurrence_dependencies):
        tag_occurrence_query = """
        UNWIND $dependencies as dependency
        MATCH (source:TagOccurrence {id: dependency.source})
        MATCH (destination:TagOccurrence {id: dependency.destination})
        MERGE (source)-[:IS_DEPENDENT {type: dependency.type}]->(destination)
        """
        self.execute_query(tag_occurrence_query, {"dependencies": tag_occurrence_dependencies})

    def execute_query(self, query, params):
        results = []
        with self.get_session() as session:
            # print("Executing query:\n", query)
            # print("with params: ", params)
            for items in session.run(query, params):
                item = items["result"]
                results.append(item)
        return results


if __name__ == "__main__":
    basic_nlp = GraphBasedNLP()
    store_tag = True
    sentences = [
        "John likes green apples",
        "Melissa picked up 3 tasty red apples",
        "That tree produces small yellow apples",
        "Small people eat the large apples",
        "Jackson likes to pick small apples",
    ]
    for idx, x in enumerate(sentences):
        basic_nlp.tokenize_and_store(x, idx, store_tag)

    # basic_nlp.tokenize_and_store( 
    # "Marie Curie received the Nobel Prize in Physic in 1903. She became the first woman to win the prize.", 1, True)
    basic_nlp.close()
