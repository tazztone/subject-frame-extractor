#!/usr/bin/env python3
"""
Startup script for Frame Extractor & Analyzer v2.0
Handles missing dependencies gracefully.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check for required dependencies and provide helpful messages."""
    missing_deps = []
    
    # Check for ML dependencies
    try:
        import ultralytics
    except ImportError:
        missing_deps.append("ultralytics (YOLO)")
    
    try:
        import insightface
    except ImportError:
        missing_deps.append("insightface")
    
    try:
        import pyiqa
    except ImportError:
        missing_deps.append("pyiqa")
    
    try:
        import imagehash
    except ImportError:
        missing_deps.append("imagehash")
    
    if missing_deps:
        print("⚠️  Missing ML dependencies detected:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n📝 Note: The application will start but ML features will not work.")
        print("   Install missing dependencies with: pip install ultralytics insightface pyiqa imagehash")
        print("\n🚀 Starting application in limited mode...\n")
    else:
        print("✅ All dependencies found. Starting full application...\n")

def main():
    """Main startup function."""
    try:
        # Check dependencies
        check_dependencies()
        
        # Import and start the application
        from app.composition import CompositionRoot
        
        # Initialize composition root
        composition = CompositionRoot()
        
        # Get the UI and launch
        app_ui = composition.get_app_ui()
        demo = app_ui.build_ui()
        
        print("🎬 Frame Extractor & Analyzer v2.0")
        print("🚀 Starting application...")
        
        # Launch the Gradio interface
        demo.launch()
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n💡 Try installing missing dependencies:")
        print("   pip install ultralytics insightface pyiqa imagehash")
        sys.exit(1)
    finally:
        # Cleanup resources
        if 'composition' in locals():
            composition.cleanup()

if __name__ == "__main__":
    main()
