## Model deployments

Embedding models run locally via Ollama or accessed through the Azure OpenAI API.
Digests and access dates correspond to the configuration used for all reported results.

| Model | Engine | Model tag / deployment | Quantisation | Input template applied |
|---|---|---|---|---|
| BGE-M3 | Ollama 0.13.2 | `bge-m3:F16` (digest `94ae3e97bdebd0806f40d6f789ca4e87c5dbb75384fdf32400cc56cc649b7f81`) | F16 | None |
| Snowflake Arctic Embed L v2.0 | Ollama 0.13.2 | `snowflake-arctic-embed-l-v2.0:F16` (digest `32946098b12deac49436c0888fe80a7d3c9a97c30e14288ea72ed29ba055d718`) | F16 | None |
| EmbeddingGemma | Ollama 0.13.2 | `embeddinggemma:F32` (digest `3362f3a035071c1f7d8fed5ad1ed0fd7a5b541876a895d4518fa9763ef5e4d82`) | F32 | None |
| multilingual-E5-large | Ollama 0.13.2 | `multilingual-e5-large:F16` (digest `6d2cb8be76885c46bba7484d09c87efb8a3eea61e7f44bc252bcea56899ba2d6`) | F16 | None; no query/passage prefixes |
| Nomic Embed v2 | Ollama 0.13.2 | `nomic-embed-text-v2-moe:F32` (digest `e24e8ea5f1d89267be8ea9e0113edbb5c798cd2541c80c0f72acfd8ba96e80da`) | F32 | None; no search prefixes |
| Qwen3-Embedding-8B | Ollama 0.13.2 | `qwen3-embedding-8b:Q6_K` (digest `704383981a2f0118c7ed82f18a1dd80424cd184be76c2d5d12e83b3676a693be`) | Q6_K | None; no instruction prefix |
| text-embedding-ada-002 | Azure OpenAI via Prompt flow 1.18.0 | deployment `text-embedding-ada-002-2`, accessed 18 December 2025 | API (not applicable) | Not applicable |
| text-embedding-3-large | Azure OpenAI via Prompt flow 1.18.0 | deployment `text-embedding-3-large`, accessed 18 December 2025 | API (not applicable) | Not applicable |