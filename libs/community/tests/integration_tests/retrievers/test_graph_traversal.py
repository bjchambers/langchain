"""Test of Graph Traversal Retriever"""

import json
import os
import random
from contextlib import contextmanager
from typing import Any, Generator, Iterable, List

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pytest import FixtureRequest

from langchain_community.retrievers.generic_graph_traversal import (
    AstraTraversalAdapter,
    CassandraTraversalAdapter,
    ChromaTraversalAdapter,
    EagerNodeSelector,
    GenericGraphTraversalRetriever,
    OpenSearchTraversalAdapter,
    TraversalAdapter,
)
from langchain_community.vectorstores import Cassandra, OpenSearchVectorSearch

vector_store_types = [
    "astra-db",
    "cassandra",
    "chroma-db",
    "open-search",
]


def _doc_ids(docs: Iterable[Document]) -> List[str]:
    return [doc.id for doc in docs if doc.id is not None]


class ParserEmbeddings(Embeddings):
    """Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            vals = json.loads(text)
        except json.JSONDecodeError:
            return [0.0] * self.dimension
        else:
            assert len(vals) == self.dimension
            return vals


class EarthEmbeddings(Embeddings):
    def get_vector_near(self, value: float) -> List[float]:
        base_point = [value, (1 - value**2) ** 0.5]
        fluctuation = random.random() / 100.0
        return [base_point[0] + fluctuation, base_point[1] - fluctuation]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    def embed_query(self, text: str) -> list[float]:
        words = set(text.lower().split())
        if "earth" in words:
            vector = self.get_vector_near(0.9)
        elif {"planet", "world", "globe", "sphere"}.intersection(words):
            vector = self.get_vector_near(0.8)
        else:
            vector = self.get_vector_near(0.1)
        return vector


@pytest.fixture
def graph_vector_store_docs() -> list[Document]:
    """
    This is a set of Documents to pre-populate a graph vector store,
    with entries placed in a certain way.

    Space of the entries (under Euclidean similarity):

                      A0    (*)
        ....        AL   AR       <....
        :              |              :
        :              |  ^           :
        v              |  .           v
                       |   :
       TR              |   :          BL
    T0   --------------x--------------   B0
       TL              |   :          BR
                       |   :
                       |  .
                       | .
                       |
                    FL   FR
                      F0

    the query point is meant to be at (*).
    the A are bidirectionally with B
    the A are outgoing to T
    the A are incoming from F
    The links are like: L with L, 0 with 0 and R with R.
    """

    docs_a = [
        Document(id="AL", page_content="[-1, 9]", metadata={"label": "AL"}),
        Document(id="A0", page_content="[0, 10]", metadata={"label": "A0"}),
        Document(id="AR", page_content="[1, 9]", metadata={"label": "AR"}),
    ]
    docs_b = [
        Document(id="BL", page_content="[9, 1]", metadata={"label": "BL"}),
        Document(id="B0", page_content="[10, 0]", metadata={"label": "B0"}),
        Document(id="BR", page_content="[9, -1]", metadata={"label": "BR"}),
    ]
    docs_f = [
        Document(id="FL", page_content="[1, -9]", metadata={"label": "FL"}),
        Document(id="F0", page_content="[0, -10]", metadata={"label": "F0"}),
        Document(id="FR", page_content="[-1, -9]", metadata={"label": "FR"}),
    ]
    docs_t = [
        Document(id="TL", page_content="[-9, -1]", metadata={"label": "TL"}),
        Document(id="T0", page_content="[-10, 0]", metadata={"label": "T0"}),
        Document(id="TR", page_content="[-9, 1]", metadata={"label": "TR"}),
    ]
    for doc_a, suffix in zip(docs_a, ["l", "0", "r"]):
        doc_a.metadata["tag"] = f"ab_{suffix}"
        doc_a.metadata["out"] = f"at_{suffix}"
        doc_a.metadata["in"] = f"af_{suffix}"
    for doc_b, suffix in zip(docs_b, ["l", "0", "r"]):
        doc_b.metadata["tag"] = f"ab_{suffix}"
    for doc_t, suffix in zip(docs_t, ["l", "0", "r"]):
        doc_t.metadata["in"] = f"at_{suffix}"
    for doc_f, suffix in zip(docs_f, ["l", "0", "r"]):
        doc_f.metadata["out"] = f"af_{suffix}"
    return docs_a + docs_b + docs_f + docs_t


class CassandraSession:
    keyspace: str
    table_name: str
    session: Any

    def __init__(self, keyspace: str, table_name: str, session: Any):
        self.keyspace = keyspace
        self.table_name = table_name
        self.session = session


@contextmanager
def get_cassandra_session(
    keyspace: str, table_name: str, drop: bool = True
) -> Generator[CassandraSession, None, None]:
    """Initialize the Cassandra cluster and session"""
    from cassandra.cluster import Cluster

    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = [
            cp.strip()
            for cp in os.environ["CASSANDRA_CONTACT_POINTS"].split(",")
            if cp.strip()
        ]
    else:
        contact_points = None

    cluster = Cluster(contact_points)
    session = cluster.connect()

    try:
        session.execute(
            (
                f"CREATE KEYSPACE IF NOT EXISTS {keyspace}"
                " WITH replication = "
                "{'class': 'SimpleStrategy', 'replication_factor': 1}"
            )
        )
        if drop:
            session.execute(f"DROP TABLE IF EXISTS {keyspace}.{table_name}")

        # Yield the session for usage
        yield CassandraSession(
            keyspace=keyspace, table_name=table_name, session=session
        )
    finally:
        # Ensure proper shutdown/cleanup of resources
        session.shutdown()
        cluster.shutdown()


@pytest.fixture(scope="function")
def vector_store(
    request: FixtureRequest, embedding_type: str, vector_store_type: str
) -> Generator[VectorStore, None, None]:
    embeddings: Embeddings
    if embedding_type == "earth-embeddings":
        embeddings = EarthEmbeddings()
    elif embedding_type == "d2-embeddings":
        embeddings = ParserEmbeddings(dimension=2)
    else:
        msg = f"Unknown embeddings type: {embedding_type}"
        raise ValueError(msg)

    store: VectorStore
    if vector_store_type == "astra-db":
        try:
            from astrapy.authentication import StaticTokenProvider
            from dotenv import load_dotenv
            from langchain_astradb import AstraDBVectorStore

            load_dotenv()

            token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])

            store = AstraDBVectorStore(
                embedding=embeddings,
                collection_name="graph_test_collection",
                namespace="default_keyspace",
                token=token,
                api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            )
            yield store
            store.delete_collection()

        except (ImportError, ModuleNotFoundError):
            msg = (
                "to test graph-traversal with AstraDB, please"
                " install langchain-astradb and python-dotenv"
            )
            raise ImportError(msg)

    elif vector_store_type == "cassandra":
        with get_cassandra_session(
            table_name="graph_test_table", keyspace="graph_test_keyspace"
        ) as session:
            store = Cassandra(
                embedding=embeddings,
                session=session.session,
                keyspace=session.keyspace,
                table_name=session.table_name,
            )
            yield store
    elif vector_store_type == "chroma-db":
        store = Chroma(embedding_function=embeddings)
        yield store
        store.delete_collection()
    elif vector_store_type == "open-search":
        store = OpenSearchVectorSearch(
            opensearch_url="http://localhost:9200",
            index_name="graph_test_index",
            embedding_function=embeddings,
            engine="faiss",
        )
        yield store
        store.delete_index()  # store.index_name
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)


def get_adapter(vector_store: VectorStore, vector_store_type: str) -> TraversalAdapter:
    if vector_store_type == "astra-db":
        return AstraTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "cassandra":
        return CassandraTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "chroma-db":
        return ChromaTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "open-search":
        return OpenSearchTraversalAdapter(vector_store=vector_store)
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)


# this test has complex metadata fields (values with list type)
# only `astra-db`` and `open-search` can correctly handle
# complex metadata fields at this time.
@pytest.mark.parametrize("vector_store_type", ["astra-db", "open-search"])
@pytest.mark.parametrize("embedding_type", ["earth-embeddings"])
def test_traversal(
    vector_store: VectorStore,
    vector_store_type: str,
) -> None:
    greetings = Document(
        id="greetings",
        page_content="Typical Greetings",
        metadata={
            "incoming": "parent",
        },
    )

    doc1 = Document(
        id="doc1",
        page_content="Hello World",
        metadata={"outgoing": "parent", "keywords": ["greeting", "world"]},
    )

    doc2 = Document(
        id="doc2",
        page_content="Hello Earth",
        metadata={"outgoing": "parent", "keywords": ["greeting", "earth"]},
    )
    vector_store.add_documents([greetings, doc1, doc2])

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GenericGraphTraversalRetriever(
        store=vector_store_adapter,
        edges=[("outgoing", "incoming"), "keywords"],
        node_selector_factory = EagerNodeSelector.factory(),
        start_k=2,
        max_depth=2,
    )

    docs = retriever.invoke("Earth", start_k=1, depth=0)
    assert _doc_ids(docs) == ["doc2"]

    docs = retriever.invoke("Earth", depth=0)
    assert _doc_ids(docs) == ["doc2", "doc1"]

    docs = retriever.invoke("Earth", start_k=1, depth=1)
    assert set(_doc_ids(docs)) == {"doc2", "doc1", "greetings"}


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata


# the tests below use simple metadata fields
# astra-db, cassandra, chroma-db, and open-search
# can all handle simple metadata fields


class TestGraphTraversal:
    @pytest.mark.parametrize("vector_store_type", vector_store_types)
    @pytest.mark.parametrize("embedding_type", ["d2-embeddings"])
    def test_invoke_sync(
        self,
        vector_store: VectorStore,
        vector_store_type: str,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Graph traversal search on a vector store."""
        vector_store.add_documents(graph_vector_store_docs)

        vector_store_adapter = get_adapter(
            vector_store=vector_store,
            vector_store_type=vector_store_type,
        )

        retriever = GenericGraphTraversalRetriever(
            store=vector_store_adapter,
            edges=[("out", "in"), "tag"],
            node_selector_factory = EagerNodeSelector.factory(),
            max_depth=2,
            start_k=2,
        )

        docs = retriever.invoke(input="[2, 10]", depth=0)
        ss_labels = {doc.metadata["label"] for doc in docs}
        assert ss_labels == {"AR", "A0"}
        assert_document_format(docs[0])

        docs = retriever.invoke(input="[2, 10]")
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in docs}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        assert_document_format(docs[0])

    @pytest.mark.parametrize("vector_store_type", vector_store_types)
    @pytest.mark.parametrize("embedding_type", ["d2-embeddings"])
    async def test_invoke_async(
        self,
        vector_store: VectorStore,
        vector_store_type: str,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Graph traversal search on a graph store."""
        await vector_store.aadd_documents(graph_vector_store_docs)

        vector_store_adapter = get_adapter(
            vector_store=vector_store,
            vector_store_type=vector_store_type,
        )

        retriever = GenericGraphTraversalRetriever(
            store=vector_store_adapter,
            edges=[("out", "in"), "tag"],
            node_selector_factory = EagerNodeSelector.factory(),
            max_depth=2,
            start_k=2,
        )
        docs = await retriever.ainvoke(input="[2, 10]", depth=0)
        ss_labels = {doc.metadata["label"] for doc in docs}
        assert ss_labels == {"AR", "A0"}
        assert_document_format(docs[0])

        docs = await retriever.ainvoke(input="[2, 10]")
        ss_labels = {doc.metadata["label"] for doc in docs}
        assert ss_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        assert_document_format(docs[0])
