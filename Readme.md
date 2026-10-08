# Video Course RAG AI Assistant

A Retrieval-Augmented Generation (RAG) assistant for video analysis and queries. It transcribes videos, indexes subtitle chunks into vector embeddings, and pinpoints exact video numbers and timestamps to answer user questions.

---

## Features

- **Local Embeddings**: 100% offline embedding creation using Ollama (`bge-m3`).
- **Hybrid Answering Engine**:
  - **Online**: Ultra-fast responses via Groq (`openai/gpt-oss-120b`).
  - **Offline Fallback**: Automatically switches to local Ollama (`qwen3`) if internet/API is unavailable.
- **Accurate Timestamps**: Maps exact seconds to readable `MM:SS` format.
- **Hallucination Guardrails**: Cites content only when explicitly taught in the video chunks.

---

## Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install requests pandas numpy scikit-learn joblib python-dotenv openai-whisper
   ```
2. **System Tools**:
   - [FFmpeg](https://ffmpeg.org/) installed and added to PATH.
   - [Ollama](https://ollama.com/) running with required models:
     ```bash
     ollama pull bge-m3
     ollama pull qwen3
     ```
3. **Environment Setup**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## Pipeline Workflow

### 1. Convert Videos to Audio
Place your course videos in `videos/` and run:
```bash
python video_to_audio.py
```
*Outputs `.mp3` files into `audios/`.*

### 2. Transcribe Audio to Subtitle Chunks
Extract timestamped transcripts using OpenAI Whisper:
```bash
python audio_to_json.py
```
*Outputs transcript chunks into `jsons/`.*

### 3. Merge small chunks
```bash
python merged_chunks.py     
```
Combines every 5 segments into one chunk

### 4. Generate Vector Embeddings
Compute chunk embeddings using Ollama's `bge-m3` model:
```bash
python preprocess_json.py
```
*Saves the indexed vector store as `embeddings.joblib`.*

### 5. Ask Questions (Run Assistant)
Launch the hybrid RAG assistant:
```bash
python process_incoming_.py
```
- Automatically uses **Groq** when connected to the internet.
- Automatically falls back to **Ollama** if offline.
- Responses and query logs are saved to `response.txt`.

## Sample Output

**Question:**
> Where is HTML basics taught?

**Answer:**
> HTML basics are not covered in the subtitle excerpts you provided.
>
> - Video 1 – "Installing VS Code & How Websites Work" (23:28–23:31, 25:10–25:13, 23:31–23:34, 24:25–24:28) – HTML, CSS, and JavaScript are mentioned as topics the course will teach.
> - Video 3 – "Basic Structure of an HTML Website" (0:28–0:30) – HTML is briefly mentioned.
>
> These timestamps don't contain an actual explanation of HTML basics.

**Backend:** Groq (`openai/gpt-oss-120b`)

## Notes

The original course videos are not included in this repository due to copyright and file-size considerations.