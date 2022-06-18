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
            bucket_name: str = "haystack-ai",
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
            print(doc)
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

                    _doc["embedding"] = _doc["embedding"].tolist()

                    blob = bucket_client.blob(_doc["id"] + ".json")
                    blob.upload_from_string(
                        data=json.dumps(_doc),
                        content_type='application/json'
                    )

                progress_bar.update(batch_size)
        progress_bar.close()

    def update_embeddings(
            self,
            retriever,
            index: Optional[str] = None,
            update_existing_embeddings: bool = True,
            batch_size: int = 10_000
    ):

        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update
        :param update_existing_embeddings: Weaviate mandates an embedding while creating the document itself.
        This option must be always true for weaviate and it will update the embeddings for all the documents.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing WeaviateDocumentStore()")

        if update_existing_embeddings:
            logger.info(f"Updating embeddings for all {self.get_document_count(index=index)} docs ...")
        else:
            raise RuntimeError(
                "All the documents in Weaviate store have an embedding by default. Only update is allowed!")

        result = self._get_all_documents_in_index()

        for result_batch in get_batches_from_generator(result, batch_size):
            document_batch = [result_batch for hit in
                              result_batch]
            embeddings = retriever.embed_documents(document_batch)  # type: ignore
            assert len(document_batch) == len(embeddings)

            if embeddings[0].shape[0] != self.embedding_dim:
                raise RuntimeError(f"Embedding dim. of model ({embeddings[0].shape[0]})"
                                   f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                                   "Specify the arg `embedding_dim` when initializing WeaviateDocumentStore()")
            for doc, emb in zip(document_batch, embeddings):
                # Using update method to only update the embeddings, other properties will be in tact
                print(f"This is the embedding - {emb}")
                if self.similarity == "cosine": self.normalize_embedding(emb)
                bucket_client = storage.Client().get_bucket(self.bucket_name)
                blob = bucket_client.blob(doc["id"] + ".json")
                blob.upload_from_string(
                    data=json.dumps(doc),
                    content_type='application/json'
                )

    def _get_all_documents_in_index(
            self,
    ) -> Generator[dict, None, None]:
        bucket_client = storage.Client().get_bucket(self.bucket_name)
        blob = bucket_client.blob(self.bucket_name)
        data = []
        for blob in bucket_client.list_blobs():
            record = json.loads(blob.download_as_string(client=None))
            meta_data = {k: v for k, v in record.items() if k not in (self.content_field, self.embedding_field)}
            record["meta"] = meta_data
            data.append(record)
            print(data[-1])
        field_map = self._create_document_field_map()
        document_objects: List[Document] = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in
                            data]

        return document_objects


