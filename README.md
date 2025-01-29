# RAG Project

This repository contains the code for a Retrieval-Augmented Generation (RAG) project utilizing the following technologies:

- **Weaviate**: A vector database for semantic search and storage of embeddings.
- **Nebula Graph**: A graph database for managing complex relationships and querying graph-based data.

## Project Overview
The goal of this project is to build a system that combines:

1. **Vector Search**: Efficiently retrieve relevant information based on embeddings stored in Weaviate.
2. **Graph Querying**: Explore and leverage complex relationships between entities using Nebula Graph.
3. **RAG Pipeline**: Integrate the capabilities of both databases to enable effective retrieval-augmented text generation.

## Spacy
Spacy needs python -m spacy download en_core_web_sm

## Vector database
   After running `docker compose up -d` (Docker Compose v2 or later), make sure all containers, including Weaviate and any vectorizer services, are connected to the same Docker network. Use the `docker network` commands to verify and configure network settings if necessary.

## Graph database
   First clone as described https://docs.nebula-graph.io/3.8.0/2.quick-start/1.quick-start-workflow/,  then use `docker compose -f docker-compose-lite.yaml up -d`. Afterwards, make sure to add all NebulaGraph related containers to the same `docker network` as weaviate.


   
