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
import time

from haystack.schema import Document
from haystack.document_stores import BaseDocumentStore
from haystack.document_stores.base import get_batches_from_generator
from google.cloud import aiplatform_v1beta1
from google.protobuf import struct_pb2
from weaviate import client, AuthClientPassword
from weaviate import ObjectsBatchRequest
logger = logging.getLogger(__name__)


class GoogleVertexDocumentStore(BaseDocumentStore):

    def __init__(
            self,
            project_id: str = "cloud-shenanigans-kuria",
            bucket_name: str = "gs://haystack-ai2",
            region: str = "europe-west1",
            endpoint: str = "",
            network_name: str = "",
            auth_token: str = "",
            project_number: str = "",
            content_field: str = "",
            id_field: str = "id",
            embedding_field: str = "embedding",
            index: str = "id",
            progress_bar: bool = True,
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

        self.endpoint = "{}-aiplatform.googleapis.com".format(self.region)



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
            self.index: self.index,
        }

    def write_documents(
            self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
            batch_size: int = 10_000, duplicate_documents: Optional[str] = None):

        buck_name = "haystack-ai2"
        bucket = storage.Client().get_bucket(buck_name)
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
                blob = bucket.blob("testfile" + doc["id"]+".json")
                blob.upload_from_string(
                    data=json.dumps(doc),
                    content_type='application/json'
                )
                documents_to_index = []
                print("Batch Complete")

    def create_index(self):
        PARENT = "projects/{}/locations/{}".format(self.project_id, self.region)
        index_client = aiplatform_v1beta1.IndexServiceClient(
            client_options=dict(api_endpoint=self.endpoint)
        )

        DIMENSIONS = 100
        DISPLAY_NAME = "glove_100_1"

        treeAhConfig = struct_pb2.Struct(
            fields={
                "leafNodeEmbeddingCount": struct_pb2.Value(number_value=500),
                "leafNodesToSearchPercent": struct_pb2.Value(number_value=7),
            }
        )

        algorithmConfig = struct_pb2.Struct(
            fields={"treeAhConfig": struct_pb2.Value(struct_value=treeAhConfig)}
        )

        config = struct_pb2.Struct(
            fields={
                "dimensions": struct_pb2.Value(number_value=DIMENSIONS),
                "approximateNeighborsCount": struct_pb2.Value(number_value=150),
                "distanceMeasureType": struct_pb2.Value(string_value="DOT_PRODUCT_DISTANCE"),
                "algorithmConfig": struct_pb2.Value(struct_value=algorithmConfig),
            }
        )

        metadata = struct_pb2.Struct(
            fields={
                "config": struct_pb2.Value(struct_value=config),
                "contentsDeltaUri": struct_pb2.Value(string_value=self.bucket_name),
            }
        )

        ann_index_meta = {
            "display_name": DISPLAY_NAME,
            "description": "Glove 100 ANN index",
            "metadata": struct_pb2.Value(struct_value=metadata),
        }

        ann_index = index_client.create_index(parent=PARENT, index=ann_index_meta)

        while True:
            if ann_index.done():
                break
            print("Poll the operation to create index...")
            time.sleep(60)

        INDEX_RESOURCE_NAME = ann_index.result().name
        print("Index created -")
        print(INDEX_RESOURCE_NAME)

        return INDEX_RESOURCE_NAME
