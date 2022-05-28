from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haystack.nodes.retriever import BaseRetriever

import json
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Generator
from tqdm.auto import tqdm
import numpy as np

from haystack.schema import Document
from haystack.document_stores.sql import SQLDocumentStore

from haystack.document_stores import BaseDocumentStore
from haystack.document_stores.base import get_batches_from_generator
from inspect import Signature, signature

logger = logging.getLogger(__name__)


class VertexMatchingDocumentStore(BaseDocumentStore):
    def __init__(
            self,
            host: Union[str, List[str]] = "http://localhost",
            port: Union[int, List[int]] = 8080,
            timeout_config: tuple = (5, 15),
            username: str = None,
            password: str = None,
            index: str = "Document",
            embedding_dim: int = 768,
            content_field: str = "content",
            name_field: str = "name",
            similarity: str = "dot_product",
            index_type: str = "hnsw",
            custom_schema: Optional[dict] = None,
            return_embedding: bool = False,
            embedding_field: str = "embedding",
            progress_bar: bool = True,
            duplicate_documents: str = 'overwrite',
            **kwargs,
    ):

    def _create_new_index(self, vector_dim: int, metric_type, index_factory: str = "Flat", **kwargs):

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000, duplicate_documents: Optional[str] = None,
                        headers: Optional[Dict[str, str]] = None) -> None:

    def update_embeddings(
            self,
            retriever: 'BaseRetriever',
            index: Optional[str] = None,
            update_existing_embeddings: bool = True,
            filters: Optional[Dict[str, List[str]]] = None,
            batch_size: int = 10_000
    ):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to get embeddings for text
        :param index: Index name for which embeddings are to be updated. If set to None, the default self.index is used.
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,
                                           only documents without embeddings are processed. This mode can be used for
                                           incremental updating of embeddings, wherein, only newly indexed documents
                                           get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """

    def query_by_embedding(
            self,
            query_emb: np.ndarray,
            filters: Optional[Dict[str, List[str]]] = None,
            top_k: int = 10,
            index: Optional[str] = None,
            return_embedding: Optional[bool] = None,
            headers: Optional[Dict[str, str]] = None
    ) -> List[Document]:
