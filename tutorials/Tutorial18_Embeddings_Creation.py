# import time
# import grpc
# import h5py
# from google.cloud import aiplatform_v1beta1
# from google.protobuf import struct_pb2
#
# h5 = h5py.File("/Users/kuriakosemathew/haystack_vector_test_data/glove-100-angular.hdf5", "r")
# train = h5["train"]
# test = h5["test"]
#
# print(train[0])
#
# with open("/Users/kuriakosemathew/haystack_vector_test_data/glove100.json", "w") as f:
#     for i in range(len(train)):
#         f.write('{"id":"' + str(i) + '",')
#         f.write('"embedding":[' + ",".join(str(x) for x in train[i]) + "]}")
#         f.write("\n")


from haystack.document_stores.googlevertex_mini import GoogleVertexDocumentStore
import json

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/kuriakosemathew/Documents/work/haystack-creds.json"
def tutorial1_basic_qa_pipeline():

    document_store = GoogleVertexDocumentStore()
    print("Trying to load json file")
    docs = []
    counter = 0
    for line in open("/Users/kuriakosemathew/haystack_vector_test_data/glove100.json", "r"):
        doc = json.loads(line)
        doc["content"] = ""
        docs.append(doc)
        if counter == 20:
            break
        counter +=1
    print(f"Loaded {len(docs)} records into memory")
    print(docs[0].keys())
    document_store.write_documents(docs)
    print("Done..!")
    # document_store = GoogleVertexDocumentStore()
    document_store.create_index()



if __name__ == "__main__":
    tutorial1_basic_qa_pipeline()
