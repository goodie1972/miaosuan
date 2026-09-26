#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the new settings system
"""
import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_settings_basic():
    """Test basic settings functionality"""
    from miaosuan.settings import get_config, update_config_overrides
    
    # Get default config
    config = get_config()
    print(f"Default webui port: {config.webui.port}")
    print(f"Default mt4 port: {config.mt4.port}")
    print(f"Default artifacts dir: {config.paths.artifacts}")
    
    # Test overrides
    updated = update_config_overrides(
        webui__port=9999,
        mt4__port=3333,
        paths__artifacts="custom_artifacts"
    )
    
    print(f"Overridden webui port: {updated.webui.port}")
    print(f"Overridden mt4 port: {updated.mt4.port}")
    print(f"Overridden artifacts dir: {updated.paths.artifacts}")
    
    # Verify original config unchanged
    config2 = get_config()
    print(f"Original webui port after override: {config2.webui.port}")
    print(f"Original mt4 port after override: {config2.mt4.port}")
    print(f"Original artifacts dir after override: {config2.paths.artifacts}")
    
    assert config2.webui.port == 8686  # Should be back to default
    assert config2.mt4.port == 23232   # Should be back to default
    assert config2.paths.artifacts == "artifacts"  # Should be back to default
    
    print("✓ Basic settings test passed")


def test_yaml_loading():
    """Test YAML config loading"""
    from miaosuan.settings import get_settings_manager, reload_config
    
    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
webui:
  port: 7777
mt4:
  port: 8888
paths:
  artifacts: "test_artifacts"
""")
        yaml_path = f.name
    
    try:
        # Temporarily replace the config path
        manager = get_settings_manager()
        old_path = manager._config_path
        manager._config_path = Path(yaml_path)
        
        # Reload config
        config = manager.reload()
        print(f"YAML webui port: {config.webui.port}")
        print(f"YAML mt4 port: {config.mt4.port}")
        print(f"YAML artifacts dir: {config.paths.artifacts}")
        
        assert config.webui.port == 7777
        assert config.mt4.port == 8888
        assert config.paths.artifacts == "test_artifacts"
        
        print("✓ YAML loading test passed")
    finally:
        # Restore original path
        manager._config_path = old_path
        # Clean up temp file
        os.unlink(yaml_path)


def test_env_var_override():
    """Test environment variable override"""
    from miaosuan.settings import get_settings_manager, reload_config
    
    # Set environment variables
    os.environ['MIAOSUAN_WEBUI_PORT'] = '5555'
    os.environ['MIAOSUAN_MT4_PORT'] = '6666'
    os.environ['MIAOSUAN_PATHS_ARTIFACTS'] = 'env_artifacts'
    
    try:
        manager = get_settings_manager()
        config = manager.reload()
        print(f"Env webui port: {config.webui.port}")
        print(f"Env mt4 port: {config.mt4.port}")
        print(f"Env artifacts dir: {config.paths.artifacts}")
        
        assert config.webui.port == 5555
        assert config.mt4.port == 6666
        assert config.paths.artifacts == "env_artifacts"
        
        print("✓ Environment variable override test passed")
    finally:
        # Clean up environment variables
        del os.environ['MIAOSUAN_WEBUI_PORT']
        del os.environ['MIAOSUAN_MT4_PORT']
        del os.environ['MIAOSUAN_PATHS_ARTIFACTS']


if __name__ == "__main__":
    test_settings_basic()
    test_yaml_loading()
    test_env_var_override()
    print("\n🎉 All settings tests passed!")