#!/usr/bin/env python3
"""
Debug script to investigate the actual memory stats structure from hardware_manager.
"""

import sys
sys.path.insert(0, '/app')

from runner.utils.hardware_manager import hardware_manager

def debug_memory_stats():
    """Debug the memory stats structure."""
    print("🔍 Debugging Hardware Manager Memory Stats Structure...")
    
    try:
        memory_stats = hardware_manager.update_all_memory_stats()
        
        if memory_stats:
            print(f"📊 Found {len(memory_stats)} entries in memory_stats")
            
            for key, stats in memory_stats.items():
                print(f"\n🔑 Key: {key}")
                print(f"   Type: {type(stats)}")
                
                # List all attributes of the stats object
                if hasattr(stats, '__dict__'):
                    print("   Attributes:")
                    for attr_name, attr_value in stats.__dict__.items():
                        print(f"     {attr_name}: {attr_value}")
                else:
                    print("   Object attributes:")
                    for attr_name in dir(stats):
                        if not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(stats, attr_name)
                                if not callable(attr_value):
                                    print(f"     {attr_name}: {attr_value}")
                            except Exception:
                                print(f"     {attr_name}: <error accessing>")
        else:
            print("❌ No memory stats returned")
            
    except Exception as e:
        print(f"❌ Error getting memory stats: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_memory_stats()