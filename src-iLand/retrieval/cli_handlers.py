"""
CLI Handlers for iLand Retrieval System

Core CLI class and initialization handling.
"""

from typing import Dict, List, Optional, Any
from colorama import Fore, Style

from .cli_utils import (
    setup_imports, validate_api_key, print_colored_header, 
    print_success, print_error, print_warning, get_retrieval_components
)
from .cli_operations import iLandCLIOperations


class iLandRetrievalCLI:
    """Command line interface for iLand retrieval system."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.router = None
        self.adapters = {}
        self.api_key = validate_api_key()
        self.cache_manager = None
        self.parallel_executor = None
        self.response_synthesizer = None
        self.operations = None
        
        # Initialize response synthesizer for natural language responses
        self._initialize_response_synthesizer()
    
    def _initialize_response_synthesizer(self):
        """Initialize the response synthesizer for generating natural language responses."""
        if not self.api_key:
            print_warning("Response synthesis not available (missing API key)")
            return
        
        try:
            imports = setup_imports()
            synthesis_imports = imports.get('synthesis', {})
            
            if not synthesis_imports.get('get_response_synthesizer') or not synthesis_imports.get('OpenAI'):
                print_warning("Response synthesis not available (missing dependencies)")
                return
            
            llm = synthesis_imports['OpenAI'](model="gpt-4o-mini", api_key=self.api_key)
            self.response_synthesizer = synthesis_imports['get_response_synthesizer'](
                response_mode=synthesis_imports['ResponseMode'].COMPACT,
                llm=llm
            )
        except Exception as e:
            print_warning(f"Could not initialize response synthesizer: {e}")
    
    def load_embeddings(self, embedding_type: str = "all") -> bool:
        """
        Load iLand embeddings and create retriever adapters.
        
        Args:
            embedding_type: Type of embeddings to load ("all", "latest", "specific")
            
        Returns:
            True if successful, False otherwise
        """
        print_colored_header(f"Loading iLand embeddings (type: {embedding_type})")
        
        try:
            imports = setup_imports()
            embeddings_utils = imports.get('embeddings', {})
            
            if embedding_type == "all" and embeddings_utils.get('load_all'):
                embeddings_data, batch_path = embeddings_utils['load_all']()
            elif embedding_type == "latest" and embeddings_utils.get('load_latest'):
                embeddings_data, batch_path = embeddings_utils['load_latest']()
            else:
                print_error(f"Embedding type '{embedding_type}' not supported or utilities not available")
                return False
            
            if not embeddings_data:
                print_error("No embedding data loaded")
                return False
            
            print_success(f"Loaded {len(embeddings_data)} embeddings from {batch_path}")
            
            # Create adapters
            return self._create_retriever_adapters(embeddings_data)
            
        except Exception as e:
            print_error(f"Error loading embeddings: {e}")
            return False
    
    def _create_retriever_adapters(self, embeddings_data) -> bool:
        """Create retriever adapters from embedding data."""
        try:
            components = get_retrieval_components()
            adapters = components['adapters']
            
            # Create adapters for the main iLand index
            index_name = "iland_land_deeds"
            self.adapters[index_name] = {}
            
            print_colored_header("Creating retriever adapters...")
            
            # Create each adapter type
            adapter_configs = [
                ("vector", adapters['vector']),
                ("summary", adapters['summary']),
                ("metadata", adapters['metadata']),
                ("hybrid", adapters['hybrid']),
                ("planner", adapters['planner']),
            ]
            
            for adapter_name, adapter_class in adapter_configs:
                self.adapters[index_name][adapter_name] = adapter_class.from_iland_embeddings(
                    embeddings_data, api_key=self.api_key
                )
                print_success(f"{adapter_name.title()} adapter created")
            
            # Special handling for chunk decoupling adapter (requires two embeddings)
            self.adapters[index_name]["chunk_decoupling"] = adapters['chunk_decoupling'].from_iland_embeddings(
                embeddings_data, embeddings_data, api_key=self.api_key
            )
            print_success("Chunk decoupling adapter created")
            
            # Special handling for recursive adapter
            vector_index = self.adapters[index_name]["vector"].index
            self.adapters[index_name]["recursive"] = adapters['recursive'].from_iland_indices(
                vector_index, vector_index
            )
            print_success("Recursive adapter created")
            
            print_success(f"Successfully created {len(self.adapters[index_name])} adapters")
            return True
            
        except Exception as e:
            print_error(f"Error creating adapters: {e}")
            return False
    
    def create_router(self, strategy_selector: str = "llm") -> bool:
        """
        Create the iLand router retriever.
        
        Args:
            strategy_selector: Strategy selection method
            
        Returns:
            True if successful, False otherwise
        """
        if not self.adapters:
            print_error("No adapters available. Please load embeddings first.")
            return False
        
        try:
            print(f"Creating iLand router with strategy selector: {strategy_selector}")
            
            components = get_retrieval_components()
            
            # Create index classifier
            classifier = components['classifier'](api_key=self.api_key)
            
            # Create router
            self.router = components['router'](
                retrievers=self.adapters,
                index_classifier=classifier,
                strategy_selector=strategy_selector,
                api_key=self.api_key
            )
            
            # Initialize operations handler
            self.operations = iLandCLIOperations(
                router=self.router,
                adapters=self.adapters,
                response_synthesizer=self.response_synthesizer,
                cache_manager=self.cache_manager,
                parallel_executor=self.parallel_executor
            )
            
            print_success("iLand router created successfully")
            return True
            
        except Exception as e:
            print_error(f"Error creating router: {e}")
            return False
    
    def setup_performance_optimizations(self, enable_caching: bool = True, 
                                       enable_parallel: bool = True) -> bool:
        """
        Setup performance optimization features.
        
        Args:
            enable_caching: Enable query result caching
            enable_parallel: Enable parallel strategy execution
            
        Returns:
            True if successful, False otherwise
        """
        try:
            components = get_retrieval_components()
            
            # Setup cache manager
            if enable_caching:
                self.cache_manager = components['cache'].from_env()
                print_success("Cache manager initialized")
            
            # Setup parallel executor
            if enable_parallel:
                self.parallel_executor = components['parallel'](max_workers=3)
                print_success("Parallel executor initialized")
            
            # Update operations handler with new components
            if self.operations:
                self.operations.cache_manager = self.cache_manager
                self.operations.parallel_executor = self.parallel_executor
            
            return True
            
        except Exception as e:
            print_error(f"Error setting up performance optimizations: {e}")
            return False
    
    def show_batch_summary(self):
        """Show summary of available iLand embedding batches."""
        try:
            imports = setup_imports()
            batch_summary_func = imports.get('embeddings', {}).get('batch_summary')
            
            if batch_summary_func:
                summary = batch_summary_func()
                print_colored_header("iLand Embedding Batch Summary:")
                print(summary)
            else:
                print_warning("Batch summary utility not available")
        except Exception as e:
            print_error(f"Error getting batch summary: {e}")
    
    def show_cache_stats(self):
        """Show cache performance statistics."""
        if not self.cache_manager:
            print_error("Cache manager not initialized")
            return
        
        stats = self.cache_manager.get_stats()
        print_colored_header("Cache Performance Statistics:")
        
        query_stats = stats.get("query_cache", {})
        print(f"Query Cache:")
        print(f"  Hit rate: {query_stats.get('hit_rate', 0):.2%}")
        print(f"  Total queries: {query_stats.get('total_queries', 0)}")
        print(f"  Cache size: {query_stats.get('size', 0)}/{query_stats.get('max_size', 0)}")
        print(f"  TTL: {query_stats.get('ttl_seconds', 0)}s")
    
    def clear_caches(self):
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear_all_caches()
            print_success("All caches cleared")
        else:
            print_error("Cache manager not initialized")
    
    def test_retrieval_strategies(self, strategy_selector: str = "llm") -> Dict[str, Any]:
        """
        Run comprehensive retrieval strategy tests.
        
        Args:
            strategy_selector: Strategy selection method to test
            
        Returns:
            Test results
        """
        try:
            # Import the test suite
            import sys
            from pathlib import Path
            tests_dir = Path(__file__).parent.parent.parent / 'tests'
            sys.path.insert(0, str(tests_dir))
            
            from test_iland_retrieval_strategies import iLandRetrievalStrategyTester
            
            print_colored_header(f"Running iLand Retrieval Strategy Tests")
            print(f"Strategy Selector: {strategy_selector}")
            
            # Create and run tester
            tester = iLandRetrievalStrategyTester()
            results = tester.run_all_tests(strategy_selector)
            
            return results
            
        except ImportError as e:
            print_error(f"Error importing test suite: {e}")
            return {}
        except Exception as e:
            print_error(f"Error running strategy tests: {e}")
            return {}
    
    # Delegate operation methods to the operations handler
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Execute a query using the iLand router."""
        if not self.operations:
            print_error("Operations handler not initialized. Please create router first.")
            return []
        return self.operations.execute_query(query_text, top_k)
    
    def test_strategies(self, test_queries: List[str], top_k: int = 3) -> Dict[str, Any]:
        """Test all strategies with a set of queries."""
        if not self.operations:
            print_error("Operations handler not initialized. Please create router first.")
            return {}
        return self.operations.test_all_strategies(test_queries, top_k)
    
    def test_parallel_strategies(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Test query using parallel strategy execution."""
        if not self.operations:
            print_error("Operations handler not initialized. Please create router first.")
            return {}
        return self.operations.test_parallel_strategies(query, top_k)
    
    def detailed_rag_response(self, query_text: str):
        """Generate a detailed RAG response for a query."""
        if not self.operations:
            print_error("Operations handler not initialized. Please create router first.")
            return
        self.operations.generate_detailed_rag_response(query_text)
    
    def interactive_mode(self):
        """Start interactive query mode."""
        print_colored_header("iLand Retrieval Interactive Mode")
        print(f"{Fore.WHITE}Enter queries to test the retrieval system.{Style.RESET_ALL}")
        self._show_interactive_commands()
        
        while True:
            try:
                # Use colored prompt for input
                query = input(f"{Fore.LIGHTBLUE_EX}iLand> {Style.RESET_ALL}").strip()
                
                if not query:
                    continue
                elif query == "/quit":
                    print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                    break
                elif query == "/help":
                    self._show_interactive_commands()
                elif query == "/summary":
                    self.show_batch_summary()
                elif query == "/cache-stats":
                    self.show_cache_stats()
                elif query == "/clear-cache":
                    self.clear_caches()
                elif query.startswith("/strategies "):
                    test_query = query[12:]  # Remove "/strategies "
                    if test_query:
                        self.test_strategies([test_query], top_k=3)
                elif query.startswith("/parallel "):
                    test_query = query[10:]  # Remove "/parallel "
                    if test_query:
                        self.test_parallel_strategies(test_query, top_k=5)
                elif query.startswith("/response "):
                    test_query = query[10:]  # Remove "/response "
                    if test_query:
                        self.detailed_rag_response(test_query)
                else:
                    self.query(query)
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Exiting...{Style.RESET_ALL}")
                break
            except Exception as e:
                print_error(f"Error: {e}")
    
    def _show_interactive_commands(self):
        """Show available interactive commands."""
        commands = [
            "/quit - Exit interactive mode",
            "/help - Show this help",
            "/summary - Show batch summary",
            "/strategies <query> - Test query with all strategies",
            "/parallel <query> - Test parallel strategy execution",
            "/response <query> - Get detailed RAG response for query",
            "/cache-stats - Show cache statistics",
            "/clear-cache - Clear all caches"
        ]
        
        print(f"{Fore.GREEN}Commands:{Style.RESET_ALL}")
        for command in commands:
            print(f"{Fore.GREEN}  {command}{Style.RESET_ALL}")
        print() 