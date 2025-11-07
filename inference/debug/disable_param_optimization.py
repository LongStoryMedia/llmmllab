#!/usr/bin/env python3
"""
Script to disable parameter optimization for testing.
"""

import yaml
import os

def disable_parameter_optimization():
    """Disable parameter optimization in user config."""
    
    # Path to user config
    config_path = "/app/config/user_config.yaml"
    
    # Check if file exists
    if not os.path.exists(config_path):
        print("❌ User config file not found at", config_path)
        return
    
    # Read current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("📊 Current parameter optimization settings:")
    param_opt = config.get('parameter_optimization', {})
    print(f"  enabled: {param_opt.get('enabled', 'NOT SET')}")
    
    # Disable parameter optimization
    if 'parameter_optimization' not in config:
        config['parameter_optimization'] = {}
    
    config['parameter_optimization']['enabled'] = False
    
    # Write back
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    print("✅ Parameter optimization disabled!")
    print("🔄 You may need to restart the container for changes to take effect.")

if __name__ == "__main__":
    disable_parameter_optimization()