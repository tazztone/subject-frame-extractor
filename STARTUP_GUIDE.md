# Startup Guide - Frame Extractor & Analyzer v2.0

## Quick Start

The application has been successfully refactored into a modular architecture. You can now start it in several ways:

### Option 1: New Modular Way (Recommended)
```bash
python main.py
```

### Option 2: Legacy Way (Still Works)
```bash
python app.py  # Shows deprecation warning, redirects to main.py
```

### Option 3: Startup Script (With Dependency Checking)
```bash
python start_app.py
```

## What You'll See

When you start the application, you'll see:

```
WARNING: Missing ML dependencies detected:
   - ultralytics (YOLO)
   - pyiqa
   - imagehash

Note: The application will start but ML features will not work.
   Install missing dependencies with: pip install ultralytics insightface pyiqa imagehash

Starting application in limited mode...

Frame Extractor & Analyzer v2.0
Starting application...
```

## Missing Dependencies

The application will start even without ML dependencies, but some features won't work:

- **ultralytics (YOLO)**: For person detection
- **insightface**: For face analysis
- **pyiqa**: For image quality assessment
- **imagehash**: For deduplication

## Installing Full Dependencies

To get all features working, install the missing dependencies:

```bash
pip install ultralytics insightface pyiqa imagehash
```

## Architecture Benefits

The new modular architecture provides:

1. **Clean Separation**: Each module has a single responsibility
2. **Easy Testing**: Dependencies can be mocked for testing
3. **Lazy Loading**: Heavy ML dependencies only load when needed
4. **Backward Compatibility**: Old `app.py` still works
5. **Better Error Handling**: Graceful handling of missing dependencies

## File Structure

```
project/
├── main.py                    # New entry point
├── start_app.py              # Startup script with dependency checking
├── app.py                    # Legacy entry point (deprecated)
├── app/                      # Main application package
│   ├── composition.py        # Dependency injection
│   ├── core/                 # Core infrastructure
│   ├── domain/               # Business logic
│   ├── io/                   # Input/Output operations
│   ├── ml/                   # Machine learning adapters
│   ├── masking/              # Subject masking
│   ├── pipelines/            # Analysis pipelines
│   └── ui/                   # User interface
└── REFACTOR_COMPLETE.md      # Complete documentation
```

## Troubleshooting

### If the application won't start:
1. Check that FFmpeg is installed: `ffmpeg -version`
2. Install missing Python dependencies
3. Check the error messages for specific issues

### If ML features don't work:
1. Install the missing ML dependencies listed above
2. Restart the application
3. Check that CUDA is available if using GPU features

## Development

For developers working on the codebase:

- Use the new modular structure for new features
- Follow the dependency injection pattern in `CompositionRoot`
- Add new modules in appropriate directories
- Update `CompositionRoot` for new dependencies

The refactoring is complete and the application is ready for production use!
