---
title: TravelMind RAG
emoji: 🌴
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# TravelMind RAG

TravelMind RAG is a portfolio-ready Kerala itinerary planner built with
LangChain, OpenAI embeddings, ChromaDB and Gradio. It retrieves relevant travel
knowledge before generating an itinerary, so the model receives grounded
context rather than relying only on its general knowledge.

## Live demo

- Hugging Face Space: add your deployed Space URL here
- Portfolio: add your portfolio URL here

## RAG architecture

~~~text
Curated travel guide
        ↓
Recursive text chunking
        ↓
OpenAI embeddings
        ↓
Persistent ChromaDB collection
        ↓
Semantic similarity search
        ↓
Retrieved travel passages
        ↓
OpenAI chat model
        ↓
Grounded itinerary + displayed sources
~~~

## Features

- Personalised itinerary generation
- Destination, duration, budget and traveller-type inputs
- Interest-based planning
- LangChain document loading and chunking
- OpenAI text embeddings
- Persistent local ChromaDB vector storage
- Semantic similarity search
- Retrieved source display
- Gradio web interface
- Secure environment-variable configuration
- Guardrails against presenting demonstration data as live information

## Technology

- Python
- LangChain
- OpenAI API
- OpenAI Embeddings
- ChromaDB
- Retrieval-Augmented Generation
- Semantic Search
- Gradio
- Hugging Face Spaces

## Run locally

### 1. Clone the repository

~~~bash
git clone https://github.com/YOUR_USERNAME/travelmind-rag.git
cd travelmind-rag
~~~

### 2. Create a virtual environment

Windows:

~~~bash
python -m venv .venv
.venv\Scripts\activate
~~~

macOS or Linux:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 4. Set the API key

PowerShell:

~~~powershell
$env:OPENAI_API_KEY="your-key"
~~~

Command Prompt:

~~~cmd
set OPENAI_API_KEY=your-key
~~~

macOS or Linux:

~~~bash
export OPENAI_API_KEY="your-key"
~~~

Never commit the API key to GitHub.

### 5. Run the application

~~~bash
python app.py
~~~

Open the local Gradio URL shown in the terminal.

## Deploy to Hugging Face Spaces

1. Sign in to Hugging Face.
2. Create a new Space.
3. Select **Gradio** as the SDK.
4. Choose **Public** visibility for an employer-facing demo.
5. Upload all project files while preserving the data directory.
6. Open the Space **Settings** page.
7. Add OPENAI_API_KEY under **Secrets**.
8. Wait for the Space to build.
9. Test several destinations and copy the public Space URL.

Do not put the API key in the README, source code or a public variable.

## How it works

### Indexing

The application loads data/travelmind_kerala_knowledge.txt, splits it into
overlapping chunks and creates an embedding for each chunk. ChromaDB stores the
text, vectors and source metadata.

### Retrieval

The user preferences become a semantic query. ChromaDB returns the four chunks
whose embeddings are closest to the query embedding.

### Generation

The retrieved chunks are added to the prompt. The chat model creates an
itinerary while being instructed not to invent live prices, schedules, weather
or availability.

## Security and cost

- The OpenAI key is read only from OPENAI_API_KEY.
- Environment files are excluded from Git.
- Public users can trigger API calls, so configure an API usage limit.
- Input lengths and interface choices are restricted.
- Application errors do not expose secret values.

## Limitations

- The included knowledge base is demonstration data.
- It does not provide live weather, schedules, prices or bookings.
- The listed destinations are limited to the included Kerala guide.
- The Space may need time to restart after sleeping.
- Retrieved chunks should be evaluated before presenting the system as
  production-ready.

## Future improvements

- Replace sample text with verified destination documents.
- Add PDF and web-document ingestion.
- Attach destination and last-verified metadata.
- Add metadata filtering and retrieval evaluation.
- Integrate live weather and transport APIs.
- Connect the RAG backend to the full TravelMind frontend.

## LinkedIn project description

> Developed a Retrieval-Augmented Generation travel-planning application that
> creates personalised Kerala itineraries using curated destination knowledge
> and semantic retrieval. Implemented LangChain document chunking, OpenAI
> embeddings and persistent ChromaDB vector storage, then built a Gradio
> interface that displays both generated itineraries and retrieved sources.

## Disclaimer

TravelMind RAG is an educational portfolio project. Users must verify live
travel information with official sources before making decisions.

## License

MIT
