import logging
from haystack.document_stores import BaseDocumentStore
from google.cloud import storage
import hashlib
import re
import uuid
from typing import Dict, Generator, List, Optional, Union

import logging
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

from haystack.schema import Document
from haystack.document_stores import BaseDocumentStore
from haystack.document_stores.base import get_batches_from_generator

from weaviate import client, AuthClientPassword
from weaviate import ObjectsBatchRequest
logger = logging.getLogger(__name__)


class GoogleVertexDocumentStore(BaseDocumentStore):

    def __init__(
            self,
            project_id: str = "",
            bucket_name: str = "",
            region: str = "",
            endpoint: str = "",
            network_name: str = "",
            auth_token: str = "",
            project_number: str = "",
            content_field: str = "",
            embedding_field: str = "",
            progress_bar: bool = True,
    ):

        self.progress_bar = progress_bar
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        self.endpoint = endpoint
        self.network_name = network_name
        self.auth_token = auth_token
        self.project_number = project_number
        self.content_field = content_field
        self.embedding_field = embedding_field




    # def upload_documents_to_gcs(self):
    #     storage_client = storage.Client()
    #     bucket = storage_client.bucket(self.bucket_name)
    #     blob = bucket.blob(destination_blob_name)
    #
    #     blob.upload_from_filename(source_file_name)
    #
    #     print(
    #         "File {} uploaded to {}.".format(
    #             source_file_name, destination_blob_name
    #         )
    #     )

    def _create_document_field_map(self) -> Dict:
        return {
            self.content_field: "content",
            self.embedding_field: "embedding"
        }

    def write_documents(
            self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
            batch_size: int = 10_000, duplicate_documents: Optional[str] = None):

        buck_name = "gs://cloud-shenanigans-kuriaaip-2021122616102"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name=buck_name, user_project="cloud-shenanigans-kuria")

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        documents_to_index = []
        for doc in document_objects:
            print("Starting a batch")
            _doc = {
                **doc.to_dict(field_map=self._create_document_field_map())
            }  # type: Dict[str, Any]

            # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
            # we "unnest" all value within "meta"
            documents_to_index.append(_doc)

            # Pass batch_size number of documents to bulk
            if len(documents_to_index) % batch_size == 0:
                blob = storage.Blob(name="test", bucket=buck_name)
                blob.upload_from_string(
                    data=json.dumps(documents_to_index),
                    content_type='application/json',
                    client=self._client,
                )
                documents_to_index = []
                print("Batch Complete")
