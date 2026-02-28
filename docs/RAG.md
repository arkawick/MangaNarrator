# RAG — Retrieval-Augmented Generation

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by first retrieving relevant information from a knowledge base, then passing that context to the LLM along with the user's query. Instead of relying solely on the LLM's training data (which may be outdated or incomplete), RAG grounds the response in specific, up-to-date documents.

```
Query
  │
  ▼
Embedding Model → Query Vector
  │
  ▼
Vector Database (FAISS) → Retrieve top-k similar documents
  │
  ▼
LLM (Ollama / LLaMA3)
[Retrieved Context] + [Query] → Enhanced Response
```

---

## Purpose in ECHO-TOON

ECHO-TOON's RAG pipeline serves two goals:

### 1. Character Personality Enrichment

A manga character speaks differently depending on who they are:
- Kiritsugu (Fate/Zero): Cold, tactical, few words
- Shirou Emiya (Fate/Stay Night): Earnest, idealistic, verbose
- Guts (Berserk): Blunt, aggressive, minimal

If the user provides the manga's source novel, character wiki, or previous chapter summaries, RAG can retrieve character-specific personality context and pass it to the LLM — enabling narration that sounds true to each character.

### 2. Story Arc Context

The LLM doesn't know the manga's plot unless it was in training data. If the user provides chapter summaries or plot notes, RAG can retrieve relevant story context for the current panel (e.g., "this scene follows the betrayal of X in the previous chapter"), enabling richer narration.

---

## Pipeline Components

### 1. Sentence Transformers — Embedding Model

Sentence Transformers convert text into dense numerical vectors (embeddings) that capture semantic meaning. Similar sentences produce similar vectors.

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')   # Fast, lightweight (80MB)

sentences = [
    "Kiritsugu is a cold and calculating assassin.",
    "He prioritizes the mission over personal feelings.",
    "He struggles with using cruel means for noble ends."
]

embeddings = model.encode(sentences)
print(embeddings.shape)   # (3, 384)
```

### Popular Embedding Models

| Model | Dimensions | Size | Speed | Quality |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 80 MB | Very Fast | Good |
| all-mpnet-base-v2 | 768 | 420 MB | Medium | Better |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 470 MB | Fast | Good (multilingual) |
| BAAI/bge-small-en-v1.5 | 384 | 130 MB | Fast | Good |
| BAAI/bge-large-en-v1.5 | 1024 | 1.3 GB | Slow | Excellent |

For ECHO-TOON: `all-MiniLM-L6-v2` is recommended (small, fast, sufficient for character context retrieval).

---

### 2. FAISS — Vector Database

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over large collections of high-dimensional vectors.

```bash
pip install faiss-cpu   # CPU version
pip install faiss-gpu   # GPU version (CUDA required)
```

### Building a FAISS Index

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Your knowledge base (character descriptions, story summaries, etc.)
documents = [
    "Kiritsugu Emiya is the main protagonist of Fate/Zero. He is cold and ruthless.",
    "Kiritsugu uses a Contender pistol loaded with Origin Bullets made from his own ribs.",
    "Kiritsugu's Innate Time Control allows him to slow or accelerate his own time.",
    "Irisviel von Einzbern is Kiritsugu's wife and the vessel for the Holy Grail.",
    "Irisviel is warm, cheerful, and deeply loves her family despite knowing her fate.",
    "Saber, whose true identity is King Arthur, is Kiritsugu's Servant.",
    "The Fourth Holy Grail War is set in Fuyuki City in the late 1990s.",
]

# Encode all documents
doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# Normalize for cosine similarity
faiss.normalize_L2(doc_embeddings)

# Build index (Inner Product = cosine similarity after L2 normalization)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

print(f"Index contains {index.ntotal} vectors")
```

### Querying FAISS

```python
def retrieve(query, top_k=3):
    query_vector = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": documents[idx],
            "score": float(distances[0][i])
        })
    return results

# Example retrieval
results = retrieve("Who is Kiritsugu and how does he fight?")
for r in results:
    print(f"[{r['score']:.3f}] {r['text']}")
```

### Saving and Loading the Index

```python
# Save
faiss.write_index(index, "echo_toon_knowledge.faiss")

# Load
loaded_index = faiss.read_index("echo_toon_knowledge.faiss")
```

---

### 3. Document Loading — Ingesting Knowledge

#### From PDF (novel, wiki export, chapter summary)

```bash
pip install pypdf2
```

```python
from PyPDF2 import PdfReader

def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
```

#### From Plain Text

```python
def load_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()
```

#### Chunking (Critical for RAG Quality)

Never embed entire documents as single vectors. Split into smaller chunks:

```python
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Usage
raw_text = load_pdf("fate_zero_wiki.pdf")
chunks = chunk_text(raw_text, chunk_size=200, overlap=30)
```

---

## Full RAG Pipeline for ECHO-TOON

```python
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

class MangaRAG:
    def __init__(self, documents, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.documents = documents
        self.index = self._build_index()

    def _build_index(self):
        embeddings = self.embedding_model.encode(self.documents).astype('float32')
        faiss.normalize_L2(embeddings)
        idx = faiss.IndexFlatIP(embeddings.shape[1])
        idx.add(embeddings)
        return idx

    def retrieve(self, query, top_k=3):
        vec = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, top_k)
        return [self.documents[i] for i in indices[0]]

    def enhance_narration(self, character_name, raw_dialogue, ollama_model='llama3'):
        context_docs = self.retrieve(f"{character_name} personality speech style")
        context = "\n".join(context_docs)

        prompt = f"""Character context:
{context}

Given this character context, rewrite the following dialogue in the character's authentic voice.
Keep the core meaning but adjust tone, word choice, and style to match the character.

Character: {character_name}
Dialogue: "{raw_dialogue}"

Enhanced version:"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"].strip()


# Usage
documents = [
    "Kiritsugu Emiya speaks in short, measured sentences. He rarely shows emotion.",
    "Kiritsugu is pragmatic — he will use any method necessary to achieve his goal.",
    ...
]

rag = MangaRAG(documents)
enhanced = rag.enhance_narration("Kiritsugu", "We need to end this now.")
print(enhanced)
# "Now." — Kiritsugu said flatly, his eyes never leaving the target.
```

---

## Alternative Vector Stores

### ChromaDB — Easier Setup

```bash
pip install chromadb
```

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("manga_knowledge")

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

results = collection.query(query_texts=["Kiritsugu fighting style"], n_results=3)
```

ChromaDB handles embedding and indexing internally — simpler than raw FAISS but less control.

### LangChain Integration

```bash
pip install langchain langchain-community
```

```python
from langchain.vectorstores import FAISS as LC_FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_text(raw_text)

vectorstore = LC_FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

LangChain provides a higher-level abstraction but adds dependency weight.

---

## RAG Quality Tips

| Technique | Impact |
|---|---|
| Smaller chunks (100–300 words) | Better retrieval precision |
| Overlap between chunks | Preserve context at boundaries |
| L2 normalization before search | Enables cosine similarity |
| Reranking retrieved docs | Higher precision (use cross-encoder) |
| Filtering by character name | Avoids irrelevant retrievals |
| Storing metadata with chunks | Enable filtered search |

---

## Comparison Table

| Store | Ease | Speed | Scale | Local | Best For |
|---|---|---|---|---|---|
| FAISS | Medium | Very Fast | Large | Yes | Production, control |
| ChromaDB | Easy | Fast | Medium | Yes | Rapid prototyping |
| LangChain + FAISS | Easy (abstracted) | Fast | Large | Yes | Full pipeline |
| Pinecone | Easy | Very Fast | Massive | No (cloud) | Large-scale cloud |
| Weaviate | Medium | Fast | Large | Yes | Schema-based filtering |
