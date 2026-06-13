#!/usr/bin/env python3
"""
Test the z1 config path resolution fix
"""

import sys
from pathlib import Path

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def test_dualp0p3_config_resolution():
    """Test that dualp0p3 integration finds the correct config"""
    
    print("🧪 Testing DualP0P3 Config Resolution")
    print("=" * 50)
    
    try:
        from train_yolov12_dino import create_model_config_path
        
        # Test YOLOv12l with vitb16 dualp0p3
        print("✅ Testing YOLOv12l + vitb16 + dualp0p3...")
        config_path = create_model_config_path(
            yolo_size='l',
            dinoversion='3',
            dino_variant='vitb16', 
            integration='dualp0p3'
        )
        
        print(f"✅ Config path: {config_path}")
        
        # Check if it's the correct config (not dual fallback)
        if 'dualp0p3' in config_path:
            print("✅ SUCCESS: Using proper dualp0p3 config!")
            print("   Architecture: Input -> DINO3Preprocessor -> YOLOv12 -> DINO3(P3) -> Head")
            return True
        elif 'dual.yaml' in config_path:
            print("❌ FAILED: Still using dual fallback!")
            print("   This means P3+P4 integration instead of P0+P3!")
            return False
        else:
            print("❓ UNKNOWN: Unexpected config path")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_config_file_exists():
    """Check if the config file actually exists"""
    
    print("\n🧪 Checking Config File Exists")
    print("=" * 50)
    
    config_path = Path('ultralytics/cfg/models/v12/yolov12l-dino3-vitb16-dualp0p3.yaml')
    
    if config_path.exists():
        print("✅ Config file exists!")
        print(f"   Path: {config_path}")
        
        # Check if it has the right architecture
        with open(config_path, 'r') as f:
            content = f.read()
            
        has_preprocess = 'preprocess:' in content
        has_dino_preprocessor = 'DINO3Preprocessor' in content
        has_dino_backbone = 'DINO3Backbone' in content
        
        print(f"✅ Architecture Check:")
        print(f"   Preprocess section: {'✓' if has_preprocess else '✗'}")
        print(f"   DINO3Preprocessor: {'✓' if has_dino_preprocessor else '✗'}")
        print(f"   DINO3Backbone: {'✓' if has_dino_backbone else '✗'}")
        
        if has_preprocess and has_dino_preprocessor and has_dino_backbone:
            print("🎯 Perfect! True DualP0P3 architecture (P0 + P3)")
            return True
        else:
            print("❌ Missing components for true DualP0P3")
            return False
    else:
        print(f"❌ Config file missing: {config_path}")
        return False

if __name__ == "__main__":
    print("🚀 DualP0P3 Config Fix Test")
    print("=" * 60)
    
    test1_passed = test_dualp0p3_config_resolution()
    test2_passed = check_config_file_exists()
    
    print("\n📊 Test Results")
    print("=" * 60)
    print(f"Config Resolution: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Config File Check: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("🎉 SUCCESS! DualP0P3 integration is now properly configured!")
        print("\n🎯 Fixed Issues:")
        print("✅ Created missing yolov12l-dino3-vitb16-dualp0p3.yaml")
        print("✅ Fixed config fallback logic to find correct config")
        print("✅ True P0+P3 architecture instead of P3+P4 fallback")
        print("\n🚀 Your training command should now work correctly:")
        print("python train_yolov12_dino.py --data data.yaml --yolo-size l --dino-variant vitb16 --integration dualp0p3 --epochs 400")
    else:
        print("❌ Some issues remain. Check the errors above.")
        sys.exit(1)
