# RAG Project

This repository contains the code for a Retrieval-Augmented Generation (RAG) project. **This project is intended for educational purposes only.**

## How It Works
This system combines a **vector database** and a **graph database** to enhance retrieval-augmented generation. The main components include:

- **Weaviate (Vector Database)**: Stores and retrieves embeddings for semantic search.
- **Nebula Graph (Graph Database)**: Captures complex relationships between entities, enabling structured querying.
- **Turn-Based Chatting**: The system supports interactive querying, where user inputs guide iterative responses by leveraging both Weaviate and Nebula Graph for context-aware answers.

## Technologies Used
- **Weaviate**: A vector database for semantic search and storage of embeddings.
- **Nebula Graph**: A graph database for managing complex relationships and querying graph-based data.

## Project Overview
The goal of this project is to build a system that combines:

1. **Vector Search**: Efficiently retrieve relevant information based on embeddings stored in Weaviate.
2. **Graph Querying**: Explore and leverage complex relationships between entities using Nebula Graph.
3. **RAG Pipeline**: Integrate the capabilities of both databases to enable effective retrieval-augmented text generation.
4. **Turn-Based Chatting**: Facilitate an interactive retrieval experience with dynamic responses based on vector and graph data.

## Vector Database
After running `docker compose up -d` (Docker Compose v2 or later), make sure all containers, including Weaviate and any vectorizer services, are connected to the same Docker network. Use the `docker network` commands to verify and configure network settings if necessary.

## Graph Database
First, clone as described in [Nebula Graph Quick Start Guide](https://docs.nebula-graph.io/3.8.0/2.quick-start/1.quick-start-workflow/), then use:
```bash
 docker compose -f docker-compose-lite.yaml up -d
```
Afterwards, ensure all Nebula Graph-related containers are added to the same `docker network` as Weaviate.

# RAG Project

This repository contains the code for a Retrieval-Augmented Generation (RAG) project. **This project is intended for educational purposes only.**

## How It Works
This system combines a **vector database** and a **graph database** to enhance retrieval-augmented generation. The main components include:

- **Weaviate (Vector Database)**: Stores and retrieves embeddings for semantic search.
- **Nebula Graph (Graph Database)**: Captures complex relationships between entities, enabling structured querying.
- **Turn-Based Chatting**: The system supports interactive querying, where user inputs guide iterative responses by leveraging both Weaviate and Nebula Graph for context-aware answers.

## Technologies Used
- **Weaviate**: A vector database for semantic search and storage of embeddings.
- **Nebula Graph**: A graph database for managing complex relationships and querying graph-based data.

## Project Overview
The goal of this project is to build a system that combines:

1. **Vector Search**: Efficiently retrieve relevant information based on embeddings stored in Weaviate.
2. **Graph Querying**: Explore and leverage complex relationships between entities using Nebula Graph.
3. **RAG Pipeline**: Integrate the capabilities of both databases to enable effective retrieval-augmented text generation.
4. **Turn-Based Chatting**: Facilitate an interactive retrieval experience with dynamic responses based on vector and graph data.

## Vector Database
After running `docker compose up -d` (Docker Compose v2 or later), make sure all containers, including Weaviate and any vectorizer services, are connected to the same Docker network. Use the `docker network` commands to verify and configure network settings if necessary.

## Graph Database
First, clone as described in [Nebula Graph Quick Start Guide](https://docs.nebula-graph.io/3.8.0/2.quick-start/1.quick-start-workflow/), then use:
```bash
 docker compose -f docker-compose-lite.yaml up -d
```
Afterwards, ensure all Nebula Graph-related containers are added to the same `docker network` as Weaviate.

## Future Plans
- **Agentic Behavior**: Implement the ability to decompose multi-topic queries into separate sub-queries for more precise retrieval.
- **Basic Internet Searching**: Integrate a web search capability to supplement retrieved data with up-to-date external information.
