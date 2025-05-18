import argparse
import logging
import sys # For printing results to stdout

from .core.config import logger # Use the configured logger
from .pipelines.baseline_rag import run_ingestion_pipeline, get_baseline_rag_pipeline # Import pipeline functions

def main():
    parser = argparse.ArgumentParser(description="RAG Lang Reference CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector store")
    ingest_parser.add_argument("source", help="Path to the document source directory or URL")
    # Optional arguments for persistence, defaulting to config values
    ingest_parser.add_argument("--persist-dir", help="Directory to save the vector store", default=None)
    ingest_parser.add_argument("--collection", help="Name of the collection in the vector store", default=None)

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question to the RAG system")
    query_parser.add_argument("question", help="The question to ask")
    # Optional: Add arguments to override retriever search type/kwargs later

    args = parser.parse_args()

    if args.command == "ingest":
        logger.info(f"Starting ingestion from source: {args.source}")
        # Prepare optional arguments for the pipeline
        kwargs = {}
        if args.persist_dir:
            kwargs['persist_directory'] = args.persist_dir
        if args.collection:
            kwargs['collection_name'] = args.collection
            
        # Call ingestion pipeline function
        try:
            # Pass force_reload=True if needed after ingest in the same CLI call, 
            # but typically query is a separate call, so retriever reloads on next run.
            run_ingestion_pipeline(args.source, **kwargs) 
        except Exception as e:
            logger.critical(f"Ingestion failed: {e}", exc_info=True)
            # Exit with a non-zero code to indicate failure
            exit(1)
            
    elif args.command == "query":
        logger.info(f"Received query: '{args.question}'")
        try:
            # Get the RAG query pipeline
            # force_reload=False by default, uses cache unless ingest happened in same run
            rag_pipeline = get_baseline_rag_pipeline() 
            
            if rag_pipeline is None:
                logger.error("Failed to initialize the RAG pipeline. Exiting.")
                exit(1)
                
            logger.info("Invoking RAG pipeline...")
            response = rag_pipeline.invoke(args.question)
            
            logger.info("Received response from RAG pipeline.")
            # Print the response directly to stdout
            print("\nAnswer:")
            print(response)
            
        except FileNotFoundError:
            logger.error("Failed to execute query: Vector store not found. Please run ingestion first.")
            exit(1)
        except ValueError as ve:
             logger.error(f"Failed to execute query due to configuration error: {ve}")
             exit(1)
        except Exception as e:
            logger.critical(f"Query failed: {e}", exc_info=True)
            exit(1)
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 