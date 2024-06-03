# -*- coding: utf-8 -*-
"""rag no image

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g-0lRIlfei07qW895CPG_2c4KoMV_gXM
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install --upgrade langchain-together

from langchain_together.embeddings import TogetherEmbeddings

Together_key = "bf2664603dc659dc6f4c81c3ebee8edab35340390f39f8af6c450a2457214a5b"
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval", api_key=Together_key
)
index_name = "mammogram-index"

# pip install langchain-pinecone (added to ragreq-requirements.yml)
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = "88e446de-82c0-4c76-b9c9-5fb56662f003"
os.environ["PINECONE_API_KEY"] = api_key

# configure client
pc = Pinecone(api_key=api_key)
index = pc.Index("mammogram-index")
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

# index.describe_index_stats()- more useful for notebook than for .py file

query = "Is dairy (milk) linked to a higher risk of breast cancer? "
docs = docsearch.similarity_search(query, k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

fullquery = f"""Based on the image, answer this query:\n
        Query: This is the input string: {query} \n
        Here is some context, 1: {docs[0].page_content[:300]} \n
        Here is some context, 2: {docs[1].page_content[:300]}"""
