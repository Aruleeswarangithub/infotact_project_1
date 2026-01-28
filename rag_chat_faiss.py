from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)


llm = Ollama(model="mistral")   # or llama3

SYSTEM_PROMPT = """
Answer only from the given context.
If answer not found, say:
I don't know. This is outside my scope.
"""

def ask(q):
    docs = db.similarity_search(q, k=3)
    context = "\n".join([d.page_content for d in docs])
    prompt = SYSTEM_PROMPT + "\nContext:\n" + context + "\nQuestion: " + q
    return llm.invoke(prompt)

while True:
    q = input("Ask question: ")
    print(ask(q))

