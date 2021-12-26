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

def tutorial1_basic_qa_pipeline():

    document_store = GoogleVertexDocumentStore()
    print("Trying to load json file")
    docs = []
    for line in open("/Users/kuriakosemathew/haystack_vector_test_data/glove100.json", "r"):
        docs.append(json.loads(line))
    print(f"Loaded {len(docs)} records into memory")
    document_store.write_documents(docs)
    print("Done..!")


if __name__ == "__main__":
    tutorial1_basic_qa_pipeline()
