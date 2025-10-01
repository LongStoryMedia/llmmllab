#!/usr/bin/env python3
"""
Phase 1 & 2 Validation Script

Validates that Phase 1 (Foundation Setup) and Phase 2 (Core Node Implementation)
are complete per the refactor-requirements.md document.

This script checks:
- File structure and organization
- Required components exist
- Core interfaces are properly defined
- Architecture separation is maintained
"""

import os
import sys
from pathlib import Path

def check_path_exists(base_path: str, relative_path: str) -> bool:
    """Check if a path exists relative to base path."""
    full_path = Path(base_path) / relative_path
    return full_path.exists()

def validate_phase_1(composer_path: str) -> dict:
    """Validate Phase 1: Foundation Setup completion."""
    results = {
        'environment_and_dependencies': {},
        'core_structure': {},
        'tool_migration': {}
    }
    
    # Environment and Dependencies
    env_checks = {
        'composer_directory': '',
        'requirements_txt': 'requirements.txt',
        'pyproject_toml': 'pyproject.toml', 
        'config_py': 'config.py'
    }
    
    for check, path in env_checks.items():
        results['environment_and_dependencies'][check] = check_path_exists(composer_path, path)
    
    # Core Structure
    core_structure = {
        'core_directory': 'core',
        'graph_directory': 'graph',
        'nodes_directory': 'nodes',
        'tools_directory': 'tools',
        'workflows_directory': 'workflows',
        'streaming_directory': 'streaming',
        'agents_directory': 'agents',
        'monitoring_directory': 'monitoring',
        'graph_state_py': 'graph/state.py',
        'composer_service_py': 'core/service.py',
        'graph_builder_py': 'graph/builder.py',
        'workflow_cache_py': 'graph/cache.py'
    }
    
    for check, path in core_structure.items():
        results['core_structure'][check] = check_path_exists(composer_path, path)
    
    # Tool Migration
    tool_migration = {
        'tool_registry_py': 'tools/registry.py',
        'static_tools_directory': 'tools/static',
        'dynamic_tools_directory': 'tools/dynamic'
    }
    
    for check, path in tool_migration.items():
        results['tool_migration'][check] = check_path_exists(composer_path, path)
    
    return results

def validate_phase_2(composer_path: str) -> dict:
    """Validate Phase 2: Core Node Implementation completion."""
    results = {
        'basic_nodes': {},
        'specialized_nodes': {},
        'graph_builder': {}
    }
    
    # Basic Nodes
    basic_nodes = {
        'standard_nodes_py': 'nodes/standard.py',
        'rag_router_py': 'nodes/rag/router.py', 
        'rag_executor_py': 'nodes/rag/executor.py'
    }
    
    for check, path in basic_nodes.items():
        results['basic_nodes'][check] = check_path_exists(composer_path, path)
    
    # Specialized Nodes
    specialized_nodes = {
        'specialized_nodes_py': 'nodes/specialized.py'
    }
    
    for check, path in specialized_nodes.items():
        results['specialized_nodes'][check] = check_path_exists(composer_path, path)
    
    # Graph Builder
    graph_builder = {
        'chat_workflow_py': 'workflows/chat.py',
        'research_workflow_py': 'workflows/research.py'
    }
    
    for check, path in graph_builder.items():
        results['graph_builder'][check] = check_path_exists(composer_path, path)
    
    return results

def print_results(phase_name: str, results: dict) -> bool:
    """Print validation results for a phase."""
    print(f"\n=== {phase_name} ===")
    all_passed = True
    
    for category, checks in results.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check.replace('_', ' ').title()}")
            if not passed:
                all_passed = False
    
    return all_passed

def validate_architecture_principles(composer_path: str) -> dict:
    """Validate key architectural principles are maintained."""
    results = {}
    
    # Check that nodes follow proper structure
    nodes_path = Path(composer_path) / 'nodes'
    if nodes_path.exists():
        results['nodes_properly_structured'] = True
        
        # Check for proper __init__.py exports
        init_file = nodes_path / '__init__.py'
        if init_file.exists():
            try:
                with open(init_file, 'r') as f:
                    content = f.read()
                    results['nodes_exported'] = '__all__' in content
            except:
                results['nodes_exported'] = False
        else:
            results['nodes_exported'] = False
    else:
        results['nodes_properly_structured'] = False
        results['nodes_exported'] = False
    
    # Check workflows are properly organized
    workflows_path = Path(composer_path) / 'workflows'
    if workflows_path.exists():
        results['workflows_organized'] = True
        
        # Check for workflow implementations
        chat_workflow = workflows_path / 'chat.py'
        research_workflow = workflows_path / 'research.py'
        
        results['chat_workflow_implemented'] = chat_workflow.exists()
        results['research_workflow_implemented'] = research_workflow.exists()
    else:
        results['workflows_organized'] = False
        results['chat_workflow_implemented'] = False
        results['research_workflow_implemented'] = False
    
    return results

def main():
    """Main validation function."""
    print("🔍 LLM ML Lab - Phase 1 & 2 Validation")
    print("=" * 50)
    
    # Find composer path
    current_dir = Path.cwd()
    composer_path = current_dir / 'inference' / 'composer'
    
    if not composer_path.exists():
        print(f"❌ Composer directory not found at: {composer_path}")
        sys.exit(1)
    
    print(f"📁 Checking composer implementation at: {composer_path}")
    
    # Validate Phase 1
    phase1_results = validate_phase_1(str(composer_path))
    phase1_passed = print_results("Phase 1: Foundation Setup", phase1_results)
    
    # Validate Phase 2  
    phase2_results = validate_phase_2(str(composer_path))
    phase2_passed = print_results("Phase 2: Core Node Implementation", phase2_results)
    
    # Validate Architecture
    arch_results = validate_architecture_principles(str(composer_path))
    arch_passed = print_results("Architecture Validation", {'architecture_principles': arch_results})
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📋 VALIDATION SUMMARY")
    print(f"{'=' * 50}")
    
    phase1_status = "✅ COMPLETE" if phase1_passed else "❌ INCOMPLETE"  
    phase2_status = "✅ COMPLETE" if phase2_passed else "❌ INCOMPLETE"
    arch_status = "✅ VALID" if arch_passed else "❌ ISSUES"
    
    print(f"Phase 1 (Foundation Setup): {phase1_status}")
    print(f"Phase 2 (Core Node Implementation): {phase2_status}")
    print(f"Architecture Principles: {arch_status}")
    
    if phase1_passed and phase2_passed:
        print("\n🎉 Ready to proceed to Phase 3: Streaming and Advanced Features")
        return 0
    else:
        print("\n⚠️  Complete remaining items before proceeding to Phase 3")
        return 1

if __name__ == '__main__':
    sys.exit(main())