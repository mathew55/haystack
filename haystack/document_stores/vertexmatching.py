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
from google.cloud import storage

logger = logging.getLogger(__name__)


class VertexMatchingDocumentStore(BaseDocumentStore):

    def __init__(
            self,
            project_id: str = "cloud-shenanigans-kuria",
            bucket_name: str = "haystack-ai2",
            region: str = "europe-west1",
            endpoint: str = "",
            network_name: str = "",
            auth_token: str = "",
            project_number: str = "",
            content_field: str = "content",
            id_field: str = "id",
            duplicate_documents: str = 'overwrite',
            embedding_field: str = "embedding",
            embedding_dim: int = 2,
            index: str = "id",
            progress_bar: bool = True,
            similarity: str = "dot_product"
    ):
        self.index = index
        self.progress_bar = progress_bar
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region

        self.network_name = network_name
        self.auth_token = auth_token
        self.project_number = project_number
        self.content_field = content_field
        self.embedding_field = embedding_field
        self.id_field = id_field
        self.duplicate_documents = duplicate_documents
        self.endpoint = "{}-aiplatform.googleapis.com".format(self.region)
        self.embedding_dim = embedding_dim
        self.similarity = similarity

    # def _create_new_index(self, vector_dim: int, metric_type, index_factory: str = "Flat", **kwargs):

    def _create_document_field_map(self) -> Dict:
        return {
            self.content_field: "content",
            self.embedding_field: "embedding"
        }

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000, duplicate_documents: Optional[str] = None,
                        headers: Optional[Dict[str, str]] = None) -> None:
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
                          them right away in FAISS. If not, you can later call update_embeddings() to create & index them.
        :param index: (SQL) index name for storing the docs and metadata
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """
        if headers:
            raise NotImplementedError("FAISSDocumentStore does not support headers.")

        index = index or self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert duplicate_documents in self.duplicate_documents_options, \
            f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        # if not self.faiss_indexes.get(index):
        #     self.faiss_indexes[index] = self._create_new_index(
        #         vector_dim=self.vector_dim,
        #         index_factory=self.faiss_index_factory_str,
        #         metric_type=faiss.METRIC_INNER_PRODUCT,
        #     )

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in
                            documents]
        document_objects = self._handle_duplicate_documents(documents=document_objects,
                                                            index=index,
                                                            duplicate_documents=duplicate_documents)

        # Weaviate requires that documents contain a vector in order to be indexed. These lines add a
        # dummy vector so that indexing can still happen
        dummy_embed_warning_raised = False
        for doc in document_objects:
            if doc.embedding is None:
                dummy_embedding = np.random.rand(self.embedding_dim).astype(np.float32)
                doc.embedding = dummy_embedding
                if not dummy_embed_warning_raised:
                    logger.warning("No embedding found in Document object being written into Weaviate. A dummy "
                                   "embedding is being supplied so that indexing can still take place. This "
                                   "embedding should be overwritten in order to perform vector similarity searches.")
                    dummy_embed_warning_raised = True

        batched_documents = get_batches_from_generator(document_objects, batch_size)
        bucket_client = storage.Client().get_bucket(self.bucket_name)

        with tqdm(total=len(document_objects), disable=not self.progress_bar, position=0,
                  desc="Writing Documents") as progress_bar:

            for document_batch in batched_documents:
                # docs_batch = ObjectsBatchRequest()
                for idx, doc in enumerate(document_batch):
                    _doc = {
                        **doc.to_dict(field_map=self._create_document_field_map())
                    }
                    _ = _doc.pop("score", None)

                    # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
                    # we "unnest" all value within "meta"
                    if "meta" in _doc.keys():
                        for k, v in _doc["meta"].items():
                            _doc[k] = v
                        _doc.pop("meta")

                    # doc_id = str(_doc.pop("id"))
                    # vector = _doc.pop(self.embedding_field)
                    _doc["embedding"] = _doc["embedding"].tolist()
                    print(f"Original - {doc.to_dict()}")
                    print(f"Cleaned - {_doc}")

                    blob = bucket_client.blob(_doc["id"] + ".json")
                    blob.upload_from_string(
                        data=json.dumps(_doc),
                        content_type='application/json'
                    )

                    # if self.similarity == "cosine": self.normalize_embedding(vector)


                    # rename as weaviate doesn't like "_" in field names

                    # Converting content to JSON-string as Weaviate doesn't allow other nested list for tables

                    # # Check if additional properties are in the document, if so,
                    # # append the schema with all the additional properties
                    # missing_props = self._check_document(current_properties, _doc)
                    # if missing_props:
                    #     for property in missing_props:
                    #         self._update_schema(property, index)
                    #         current_properties.append(property)

                #     docs_batch.add(_doc, class_name=index, uuid=doc_id, vector=vector)
                #
                # # Ingest a batch of documents
                # results = self.weaviate_client.batch.create(docs_batch)
                # # Weaviate returns errors for every failed document in the batch
                # if results is not None:
                #     for result in results:
                #         if 'result' in result and 'errors' in result['result'] \
                #                 and 'error' in result['result']['errors']:
                #             for message in result['result']['errors']['error']:
                #                 logger.error(f"{message['message']}")
                progress_bar.update(batch_size)
        progress_bar.close()


def write_documents_old(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000, duplicate_documents: Optional[str] = None,
                        headers: Optional[Dict[str, str]] = None) -> None:
    bucket_client = storage.Client().get_bucket(self.bucket_name)
    field_map = self._create_document_field_map()

    document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
    print(f"Document index - {document_objects[0]}")
    documents_to_index = []
    for doc in document_objects:
        print("Starting a batch")
        _doc = {
            **doc.to_dict(field_map=self._create_document_field_map())
        }  # type: Dict[str, Any]

        print(f"Type of doc - {_doc}")
        # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
        # we "unnest" all value within "meta"
        documents_to_index.append(_doc)

        # Pass batch_size number of documents to bulk
        for doc in documents_to_index:
            doc["embedding"] = doc["embedding"].tolist()
            blob = bucket_client.blob("testfile" + doc["id"] + ".json")
            blob.upload_from_string(
                data=json.dumps(doc),
                content_type='application/json'
            )
            documents_to_index = []
            print("Batch Complete")


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

# def query_by_embedding(
#         self,
#         query_emb: np.ndarray,
#         filters: Optional[Dict[str, List[str]]] = None,
#         top_k: int = 10,
#         index: Optional[str] = None,
#         return_embedding: Optional[bool] = None,
#         headers: Optional[Dict[str, str]] = None
# ) -> List[Document]:
