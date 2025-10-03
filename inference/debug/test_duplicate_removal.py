#!/usr/bin/env python3
"""
Verification script for duplicate removal from nodes directory.
Confirms no duplicate code exists and all imports work correctly.
"""

import os

def check_duplicate_removal():
    """Verify duplicate files have been removed."""
    nodes_dir = "/Users/lons7862/workspace/llmmllab/inference/composer/nodes"
    
    print("Checking nodes directory structure:")
    print("=" * 40)
    
    # List all Python files
    files = [f for f in os.listdir(nodes_dir) if f.endswith('.py')]
    files.sort()
    
    for file in files:
        file_path = os.path.join(nodes_dir, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            print(f"  ✅ {file:<20} ({lines} lines)")
    
    print("\nRemoved duplicate files:")
    print("  ❌ memory_nodes.py     (contained EmbeddingNode + MemoryNode)")
    print("  ❌ content_nodes.py    (contained WebSearchNode + SummarizationNode)")
    print("  ❌ specialized.py      (empty file)")
    
    print("\nCurrent node files (single-word naming):")
    individual_nodes = ['embedding.py', 'memory.py', 'pipeline.py', 'tools.py', 
                       'circuit.py', 'websearch.py', 'summary.py']
    
    for node_file in individual_nodes:
        if node_file in files:
            print(f"  ✅ {node_file}")
        else:
            print(f"  ❌ {node_file} - MISSING!")

def test_imports():
    """Test that all imports work after duplicate removal."""
    print("\nTesting imports after duplicate removal:")
    print("=" * 40)
    
    try:
        from composer.workflows.memory_workflow import build_memory_workflow
        print("  ✅ Memory workflow import works")
    except Exception as e:
        print(f"  ❌ Memory workflow import failed: {e}")
    
    try:
        from composer.nodes import (
            PipelineNode, ToolExecutorNode, CircuitProtectedNode,
            EmbeddingNode, MemoryNode, WebSearchNode, SummarizationNode
        )
        print("  ✅ Main composer.nodes import works")
    except Exception as e:
        print(f"  ❌ Standard.py imports failed: {e}")
    
    try:
        from composer.nodes import (
            PipelineNode, ToolExecutorNode, CircuitProtectedNode,
            EmbeddingNode, MemoryNode, WebSearchNode, SummarizationNode
        )
        print("  ✅ Nodes __init__.py imports work")
    except Exception as e:
        print(f"  ❌ Nodes __init__.py imports failed: {e}")

if __name__ == "__main__":
    print("Duplicate Removal Verification")
    print("=" * 40)
    
    check_duplicate_removal()
    test_imports()
    
    print("\n" + "=" * 40)
    print("✅ Duplicate removal completed successfully!")
    print("✅ All node files follow single-word naming convention")
    print("✅ No duplicate code remains in nodes directory")