"""Test of Apache Cassandra graph index class `CassandraGraphIndex`"""

import json
import os
import random
from contextlib import contextmanager
from typing import Any, Generator, Iterable, List, Optional

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.retrievers import CassandraGraphIndex
from tests.integration_tests.cache.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
    FakeEmbeddings,
)

TEST_KEYSPACE = "graph_test_keyspace"


def _result_ids(docs: Iterable[Document]) -> List[Optional[str]]:
    return [doc.id for doc in docs]


class CassandraSession:
    table_name: str
    session: Any

    def __init__(self, table_name: str, session: Any):
        self.table_name = table_name
        self.session = session


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


@pytest.fixture
def embedding_d2() -> Embeddings:
    return ParserEmbeddings(dimension=2)


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
def earth_embeddings() -> Embeddings:
    return EarthEmbeddings()


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
        # add_links(doc_a, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
        # add_links(doc_a, Link.outgoing(kind="at_example", tag=f"tag_{suffix}"))
        # add_links(doc_a, Link.incoming(kind="af_example", tag=f"tag_{suffix}"))
    for doc_b, suffix in zip(docs_b, ["l", "0", "r"]):
        doc_b.metadata["tag"] = f"ab_{suffix}"
        # add_links(doc_b, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
    for doc_t, suffix in zip(docs_t, ["l", "0", "r"]):
        doc_t.metadata["in"] = f"at_{suffix}"
        # add_links(doc_t, Link.incoming(kind="at_example", tag=f"tag_{suffix}"))
    for doc_f, suffix in zip(docs_f, ["l", "0", "r"]):
        doc_f.metadata["out"] = f"af_{suffix}"
        # add_links(doc_f, Link.outgoing(kind="af_example", tag=f"tag_{suffix}"))
    return docs_a + docs_b + docs_f + docs_t


@contextmanager
def get_cassandra_session(
    table_name: str, drop: bool = True
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
                f"CREATE KEYSPACE IF NOT EXISTS {TEST_KEYSPACE}"
                " WITH replication = "
                "{'class': 'SimpleStrategy', 'replication_factor': 1}"
            )
        )
        if drop:
            session.execute(f"DROP TABLE IF EXISTS {TEST_KEYSPACE}.{table_name}")

        # Yield the session for usage
        yield CassandraSession(table_name=table_name, session=session)
    finally:
        # Ensure proper shutdown/cleanup of resources
        session.shutdown()
        cluster.shutdown()


@pytest.fixture(scope="function")
def graph_index_angular_mmr(
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphIndex, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphIndex(
            search_type="mmr",
            search_kwargs={"fetch_k": 2},
            edges=[("outgoing", "incoming")],
            k=2,
            embedding=AngularTwoDimensionalEmbeddings(),
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


@pytest.fixture(scope="function")
def graph_index_earth_traversal(
    earth_embeddings: Embeddings,
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphIndex, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphIndex(
            search_type="traversal",
            edges=[("outgoing", "incoming"), "keywords"],
            k=2,
            embedding=earth_embeddings,
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


@pytest.fixture(scope="function")
def graph_index_fake_traversal(
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphIndex, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphIndex(
            search_type="traversal",
            search_kwargs={"depth": 0},
            edges=[("outgoing", "incoming"), "keywords"],
            k=2,
            embedding=FakeEmbeddings(),
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


@pytest.fixture(scope="function")
def graph_index_d2_mmr(
    embedding_d2: Embeddings,
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphIndex, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphIndex(
            search_type="mmr",
            search_kwargs={
                "depth": 2,
                "fetch_k": 1,
                "adjacent_k": 2,
                "lambda_mult": 0.1,
            },
            edges=[("out", "in"), "tag"],
            k=2,
            embedding=embedding_d2,
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


@pytest.fixture(scope="function")
def graph_index_d2_traversal_depth_0(
    embedding_d2: Embeddings,
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphIndex, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphIndex(
            search_type="traversal",
            search_kwargs={"depth": 0},
            edges=[("out", "in"), "tag"],
            k=2,
            embedding=embedding_d2,
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


@pytest.fixture(scope="function")
def graph_index_d2_traversal_depth_2(
    embedding_d2: Embeddings,
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphIndex, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphIndex(
            search_type="traversal",
            search_kwargs={"depth": 2},
            edges=[("out", "in"), "tag"],
            k=2,
            embedding=embedding_d2,
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


def test_mmr_traversal(graph_index_angular_mmr: CassandraGraphIndex) -> None:
    """ Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==2 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    v0 = Document(id="v0", page_content="-0.124")
    v1 = Document(id="v1", page_content="+0.127")
    v2 = Document(id="v2", page_content="+0.25")
    v3 = Document(id="v3", page_content="+1.0")

    v0.metadata["outgoing"] = "link"
    v2.metadata["incoming"] = "link"
    v3.metadata["incoming"] = "link"

    g_index = graph_index_angular_mmr
    g_index.upsert([v0, v1, v2, v3])

    results = g_index.invoke("0.0", k=2, fetch_k=2)
    assert _result_ids(results) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    results = g_index.invoke("0.0", k=2, fetch_k=2, depth=0)
    assert _result_ids(results) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    results = g_index.invoke("0.0", k=2, fetch_k=3, depth=0)
    assert _result_ids(results) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    results = g_index.invoke("0.0", k=2, score_threshold=0.2)
    assert _result_ids(results) == ["v0"]

    # with k=4 we should get all of the documents.
    results = g_index.invoke("0.0", k=4)
    assert _result_ids(results) == ["v0", "v2", "v1", "v3"]


def test_write_retrieve_keywords(
    graph_index_earth_traversal: CassandraGraphIndex,
) -> None:
    greetings = Document(
        id="greetings",
        page_content="Typical Greetings",
        metadata={
            "incoming": "parent",
        },
    )

    node1 = Document(
        id="doc1",
        page_content="Hello World",
        metadata={"outgoing": "parent", "keywords": ["greeting", "world"]},
    )

    node2 = Document(
        id="doc2",
        page_content="Hello Earth",
        metadata={"outgoing": "parent", "keywords": ["greeting", "earth"]},
    )

    g_index = graph_index_earth_traversal
    g_index.upsert(items=[greetings, node1, node2])

    # Doc2 is more similar, but World and Earth are similar enough that doc1 also
    # shows up.
    results: Iterable[Document] = g_index.invoke("Earth", k=2)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = g_index.invoke("Earth", k=1, depth=0)
    assert _result_ids(results) == ["doc2"]

    results = g_index.invoke("Earth", k=2, depth=0)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = g_index.invoke("Earth", k=2, depth=1)
    assert _result_ids(results) == ["doc2", "doc1", "greetings"]

    # K=1 only pulls in doc2 (Hello Earth)
    results = g_index.invoke("Earth", k=1, depth=0)
    assert _result_ids(results) == ["doc2"]

    # K=1 only pulls in doc2 (Hello Earth). Depth=1 traverses to parent and via
    # keyword edge.
    results = g_index.invoke("Earth", k=1, depth=1)
    assert set(_result_ids(results)) == {"doc2", "doc1", "greetings"}


def test_metadata(graph_index_fake_traversal: CassandraGraphIndex) -> None:
    doc_a = Document(
        id="a",
        page_content="A",
        metadata={"other": "some other field", "in-link": "http://a", "tags": ["foo"]},
    )

    g_index = graph_index_fake_traversal
    g_index.upsert([doc_a])
    results = g_index.invoke("A")
    assert len(results) == 1
    assert results[0].id == "a"
    metadata = results[0].metadata
    assert metadata["other"] == "some other field"
    assert metadata["in-link"] == "http://a"
    assert metadata["tags"] == ["foo"]


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata


class TestCassandraGraphIndex:
    def test_gvs_similarity_search_sync(
        self,
        graph_index_d2_traversal_depth_0: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector g_index."""
        g_index = graph_index_d2_traversal_depth_0
        g_index.upsert(graph_vector_store_docs)
        ss_response = g_index.invoke(input="[2, 10]")
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        assert_document_format(ss_response[0])

    async def test_gvs_similarity_search_async(
        self,
        graph_index_d2_traversal_depth_0: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector store."""
        g_index = graph_index_d2_traversal_depth_0
        await g_index.aupsert(graph_vector_store_docs)
        ss_response = await g_index.invoke(query="[2, 10]")
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        assert_document_format(ss_response[0])

    def test_gvs_traversal_search_sync(
        self,
        graph_index_d2_traversal_depth_2: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_index = graph_index_d2_traversal_depth_2
        g_index.upsert(graph_vector_store_docs)
        docs: list[Document] = g_index.invoke(query="[2, 10]")
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in docs}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        assert_document_format(docs[0])

    async def test_gvs_traversal_search_async(
        self,
        graph_index_d2_traversal_depth_2: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_index = graph_index_d2_traversal_depth_2
        await g_index.aupsert(graph_vector_store_docs)
        ts_labels = set()
        async for doc in g_index.ainvoke(query="[2, 10]"):
            ts_labels.add(doc.metadata["label"])
            assert_document_format(doc)
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

    def test_gvs_mmr_traversal_search_sync(
        self,
        graph_index_d2_mmr: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        g_index = graph_index_d2_mmr
        g_index.upsert(graph_vector_store_docs)
        docs = g_index.invoke(input="[2, 10]")
        # TODO: can this rightfully be a list (or must it be a set)?
        mt_labels = {doc.metadata["label"] for doc in docs}
        assert mt_labels == {"AR", "BR"}
        assert docs[0].metadata
        assert_document_format(docs[0])

    async def test_gvs_mmr_traversal_search_async(
        self,
        graph_index_d2_mmr: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        g_index = graph_index_d2_mmr
        await g_index.aupsert(graph_vector_store_docs)
        mt_labels = set()
        async for doc in g_index.ainvoke(input="[2, 10]"):
            mt_labels.add(doc.metadata["label"])
        # TODO: can this rightfully be a list (or must it be a set)?
        assert mt_labels == {"AR", "BR"}
        assert_document_format(doc)

    def test_gvs_metadata_search_sync(
        self,
        graph_index_d2_mmr: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Metadata search on a graph vector store."""
        g_index = graph_index_d2_mmr
        g_index.upsert(graph_vector_store_docs)
        mt_response = g_index._metadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        assert doc.metadata["in"] == "at_0"
        # links = doc.metadata["links"]
        # assert len(links) == 1
        # link: Link = links.pop()
        # assert isinstance(link, Link)
        # assert link.direction == "in"
        # assert link.kind == "at_example"
        # assert link.tag == "tag_0"
        assert_document_format(doc)

    async def test_gvs_metadata_search_async(
        self,
        graph_index_d2_mmr: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Metadata search on a graph vector store."""
        g_index = graph_index_d2_mmr
        await g_index.aupsert(graph_vector_store_docs)
        mt_response = await g_index._ametadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        assert doc.metadata["in"] == "at_0"
        # links: set[Link] = doc.metadata["links"]
        # assert len(links) == 1
        # link: Link = links.pop()
        # assert isinstance(link, Link)
        # assert link.direction == "in"
        # assert link.kind == "at_example"
        # assert link.tag == "tag_0"
        assert_document_format(doc)

    def test_gvs_get_by_document_id_sync(
        self,
        graph_index_d2_mmr: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Get by document_id on a graph vector store."""
        g_index = graph_index_d2_mmr
        g_index.upsert(graph_vector_store_docs)
        docs = g_index.get(["FL"])
        assert len(docs) == 1
        doc: Document = docs[0]
        assert doc.page_content == "[1, -9]"
        assert doc.metadata["out"] == "af_l"
        # links = doc.metadata["links"]
        # assert len(links) == 1
        # link: Link = links.pop()
        # assert isinstance(link, Link)
        # assert link.direction == "out"
        # assert link.kind == "af_example"
        # assert link.tag == "tag_l"
        assert_document_format(doc)

        invalid_docs = g_index.get(["invalid"])
        assert len(invalid_docs) == 0

    async def test_gvs_get_by_document_id_async(
        self,
        graph_index_d2_mmr: CassandraGraphIndex,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Get by document_id on a graph vector store."""
        g_index = graph_index_d2_mmr
        await g_index.aupsert(graph_vector_store_docs)
        docs = await g_index.aget(["FL"])
        assert len(docs) == 1
        doc: Document = docs[0]
        assert doc.page_content == "[1, -9]"
        assert doc.metadata["out"] == "af_l"
        # links = doc.metadata["links"]
        # assert len(links) == 1
        # link: Link = links.pop()
        # assert isinstance(link, Link)
        # assert link.direction == "out"
        # assert link.kind == "af_example"
        # assert link.tag == "tag_l"
        assert_document_format(doc)

        invalid_docs = await g_index.aget(["invalid"])
        assert len(invalid_docs) == 0