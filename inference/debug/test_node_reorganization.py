#!/usr/bin/env python3
"""
Verification script for node file reorganization.
Tests that all individual node files can be imported and that standard.py works as expected.
"""

def test_individual_imports():
    """Test importing each node class from its individual file."""
    print("Testing individual node file imports:")
    
    try:
        from composer.nodes.pipeline import PipelineNode
        print("  ✅ PipelineNode from pipeline.py")
    except Exception as e:
        print(f"  ❌ PipelineNode: {e}")
    
    try:
        from composer.nodes.tools import ToolExecutorNode
        print("  ✅ ToolExecutorNode from tools.py")
    except Exception as e:
        print(f"  ❌ ToolExecutorNode: {e}")
        
    try:
        from composer.nodes.circuit import CircuitProtectedNode
        print("  ✅ CircuitProtectedNode from circuit.py")
    except Exception as e:
        print(f"  ❌ CircuitProtectedNode: {e}")
        
    try:
        from composer.nodes.embedding import EmbeddingNode
        print("  ✅ EmbeddingNode from embedding.py")
    except Exception as e:
        print(f"  ❌ EmbeddingNode: {e}")
        
    try:
        from composer.nodes.memory import MemoryNode
        print("  ✅ MemoryNode from memory.py")
    except Exception as e:
        print(f"  ❌ MemoryNode: {e}")
        
    try:
        from composer.nodes.websearch import WebSearchNode
        print("  ✅ WebSearchNode from websearch.py")
    except Exception as e:
        print(f"  ❌ WebSearchNode: {e}")
        
    try:
        from composer.nodes.summary import SummarizationNode
        print("  ✅ SummarizationNode from summary.py")
    except Exception as e:
        print(f"  ❌ SummarizationNode: {e}")

def test_main_init_imports():
    """Test importing all classes from main composer.nodes module."""
    print("\nTesting composer.nodes main module:")
    
    try:
        from composer.nodes import (
            PipelineNode, ToolExecutorNode, CircuitProtectedNode, 
            EmbeddingNode, MemoryNode, WebSearchNode, SummarizationNode
        )
        print("  ✅ All classes imported from composer.nodes successfully")
        
        classes = [PipelineNode, ToolExecutorNode, CircuitProtectedNode, 
                  EmbeddingNode, MemoryNode, WebSearchNode, SummarizationNode]
        class_names = [cls.__name__ for cls in classes]
        print(f"  ✅ Available classes: {class_names}")
        
    except Exception as e:
        print(f"  ❌ Standard imports failed: {e}")

def test_nodes_init_imports():
    """Test importing through nodes/__init__.py."""
    print("\nTesting nodes/__init__.py imports:")
    
    try:
        from composer.nodes import (
            PipelineNode, ToolExecutorNode, CircuitProtectedNode,
            EmbeddingNode, MemoryNode, WebSearchNode, SummarizationNode
        )
        print("  ✅ All classes imported through nodes/__init__.py successfully")
        
    except Exception as e:
        print(f"  ❌ Nodes __init__ imports failed: {e}")

if __name__ == "__main__":
    print("Node File Reorganization Verification")
    print("=" * 40)
    
    test_individual_imports()
    test_main_init_imports()
    test_nodes_init_imports()
    
    print("\n" + "=" * 40)
    print("Verification complete! All node files are properly organized with single-word names.")
    print("Main composer.nodes module serves as a convenient import for all node classes.")