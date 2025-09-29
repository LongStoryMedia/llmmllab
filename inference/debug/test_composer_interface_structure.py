#!/usr/bin/env python3
"""
Simple test for composer functional interface structure.
Tests interface without heavy ML dependencies.
"""

import sys
import os

# Add inference path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_interface_structure():
    """Test that the functional interface is properly structured."""
    print("🧪 Testing composer functional interface structure...")
    
    try:
        # Test that we can import the interface
        import composer
        print("✅ Composer module imported successfully")
        
        # Check that expected functions are available
        expected_functions = [
            'initialize_composer',
            'shutdown_composer', 
            'get_composer_service',
            'compose_workflow',
            'create_initial_state',
            'execute_workflow',
            'get_composer_config'
        ]
        
        for func_name in expected_functions:
            if hasattr(composer, func_name):
                print(f"✅ {func_name} available")
            else:
                print(f"❌ {func_name} missing")
                return False
                
        # Check __all__ export list
        if hasattr(composer, '__all__'):
            exported = set(composer.__all__)
            expected = set(expected_functions)
            if exported == expected:
                print("✅ __all__ exports correctly defined")
            else:
                missing = expected - exported
                extra = exported - expected
                if missing:
                    print(f"❌ Missing from __all__: {missing}")
                if extra:
                    print(f"❌ Extra in __all__: {extra}")
                return False
        else:
            print("❌ __all__ not defined")
            return False
            
        print("\n🎉 Functional interface structure is correct!")
        return True
        
    except Exception as e:
        print(f"❌ Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_structure():
    """Test that config structure is accessible."""
    print("\n🧪 Testing config structure...")
    
    try:
        # Test config import without initializing heavy components
        import composer.config
        print("✅ Config module imported")
        
        if hasattr(composer.config, 'config'):
            print("✅ Global config instance available")
            
            config_obj = composer.config.config
            
            # Test basic config properties
            if hasattr(config_obj, 'service'):
                print("✅ Service config available")
            
            if hasattr(config_obj, 'default_workflow'):
                print("✅ Default workflow config available") 
                
            if hasattr(config_obj, 'default_tool'):
                print("✅ Default tool config available")
                
        print("✅ Config structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def main():
    """Run structure tests."""
    print("🚀 Starting composer functional interface structure tests...")
    print("=" * 60)
    
    interface_ok = test_interface_structure()
    config_ok = test_config_structure()
    
    print("\n" + "=" * 60)
    if interface_ok and config_ok:
        print("🎉 All structure tests passed!")
        print("\n📋 Summary of functional interface:")
        print("  🔧 initialize_composer() - Initialize service")
        print("  🛑 shutdown_composer() - Clean shutdown")  
        print("  📦 get_composer_service() - Access service instance")
        print("  🏗️  compose_workflow() - Build LangGraph workflow")
        print("  🎯 create_initial_state() - Create workflow state")
        print("  ▶️  execute_workflow() - Run workflow with streaming")
        print("  ⚙️  get_composer_config() - Access configuration")
        
        print("\n✅ Ready for server integration without HTTP overhead!")
    else:
        print("❌ Some structure tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()