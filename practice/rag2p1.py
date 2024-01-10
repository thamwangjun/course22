from haystack import Pipeline, Document
from haystack.components.builders import DynamicChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores import DuplicatePolicy
from qdrant_haystack import QdrantDocumentStore
from qdrant_haystack.retriever import QdrantRetriever
from txtai.pipeline import Textractor
import gradio as gr

document_store = QdrantDocumentStore(
    "http://127.0.0.1",
    recreate_index=True,
    return_embedding=True,
    wait_result_from_api=True,
    index='zarathustra-rag2a'
)


retriever = QdrantRetriever(
    document_store=document_store,
    top_k=10
)

text_embedder = SentenceTransformersTextEmbedder(
        model_name_or_path="BAAI/llm-embedder",
        prefix="Represent this query for retrieving relevant documents: "
    )

template = """
Given the following information, follow my instruction.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

My Instruction: {{ question }}
"""

prompt_builder = DynamicChatPromptBuilder(runtime_variables=["documents"])

from collections import deque
from time import sleep

class QueueIterator:
    def __init__(self):
        self.queue = deque()

    def add(self, item):
        self.queue.append(item)

    def __iter__(self):
        return self

    def __next__(self):
        retry_countdown = 60
        while retry_countdown > 0:
            popped = self.pop()
            if not popped:
                retry_countdown -= 1
                sleep(10)
            else:
                return popped
        raise StopIteration

    def pop(self):
        if self.queue:
            return self.queue.popleft()
        else:
            return False
qi = QueueIterator()

tx_box = gr.Textbox(render=True, interactive=False)

def build_rag():
    global tx_box
    llm = OpenAIChatGenerator(streaming_callback=lambda chunk: qi.add(chunk.content),
                              api_base_url="", api_key="")

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.draw("rag_pipeline.png")
    return rag_pipeline


rag_pipeline = build_rag()
embeddings_generated = False

def send(message, history, files):
    global rag_pipeline
    if not history:
        rag_pipeline = build_rag()
        embeddings_generated = False
    generate_embeddings(files)

    messages = [ChatMessage.from_user(template)]
    response = rag_pipeline.run(
        {
            "text_embedder": {"text": message},
            "prompt_builder": {
                "template_variables": {"question": message},
                "prompt_source": messages
            }
        }
    )
    return response.get('llm').get('replies')[-1].content


def generate_embeddings(files: list):
    global embeddings_generated
    if embeddings_generated:
        return
    document_embedder = SentenceTransformersDocumentEmbedder(
        model_name_or_path="BAAI/llm-embedder",
        prefix="Represent this document for retrieval: "
    )
    document_embedder.warm_up()
    documents = []
    textract = Textractor(paragraphs=True)
    for file in files:
        for paragraph in textract(file):
            if len(paragraph) > 32:
                documents.append(
                    Document(
                        meta={'name': file.name},
                        content=paragraph
                    )
                )
    document_writer = DocumentWriter(document_store = document_store)
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=DocumentCleaner(), name="cleaner")
    indexing_pipeline.add_component(instance=document_embedder, name="embedder")
    indexing_pipeline.add_component(instance=document_writer, name="writer")
    indexing_pipeline.connect("cleaner", "embedder")
    indexing_pipeline.connect("embedder", "writer")
    indexing_pipeline.draw("indexing_pipeline.png")

    indexing_pipeline.run(
        {
            "cleaner": {
                "documents": documents
            },
            "writer": {
                "policy": DuplicatePolicy.OVERWRITE
            }
        }
    )
    embeddings_generated = True


demo = gr.ChatInterface(fn=send, title="RAG2A", additional_inputs=[tx_box, gr.Files(interactive=True)])
demo.launch(inline=False, debug=True)