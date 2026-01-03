"""
Script to download and setup KataCR weights for Clash Royale detection.
"""
import os
import sys
from pathlib import Path

# URLs for KataCR model weights (Google Drive links)
KATACR_WEIGHTS = {
    "detector1": {
        "url": "https://drive.google.com/file/d/1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_/view",
        "description": "YOLOv8 detector1 v0.7.13 - Large units",
    },
    "detector2": {
        "url": "https://drive.google.com/file/d/1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD/view",  
        "description": "YOLOv8 detector2 v0.7.13 - Small units",
    },
    "card_classifier": {
        "url": "https://drive.google.com/drive/folders/1Ely1gIOEOui7uHLppeS7tLXNtdkvit07",
        "description": "ResNet card classifier",
    },
    "elixir_classifier": {
        "url": "https://drive.google.com/drive/folders/1cuqD_WQaa4uOlzSVEqLUwGmy0XNucteU",
        "description": "ResNet elixir classifier",
    },
    "policy_starformer": {
        "url": "https://drive.google.com/drive/folders/1kqE_2xDainIixf4u5YD12aqT5_LxiZwZ",
        "description": "StARformer policy model (3L)",
    },
}


def print_download_instructions():
    """Print instructions for downloading KataCR weights."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        KataCR Weight Download Instructions                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  The KataCR model weights are hosted on Google Drive and must be downloaded  ║
║  manually due to authentication requirements.                                ║
║                                                                              ║
║  REQUIRED WEIGHTS (for detection):                                           ║
║  ─────────────────────────────────                                           ║
║                                                                              ║
║  1. YOLOv8 Detector1 (large units):                                          ║
║     URL: https://drive.google.com/file/d/1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_  ║
║     Save to: ./weights/detector1.pt                                          ║
║                                                                              ║
║  2. YOLOv8 Detector2 (small units):                                          ║
║     URL: https://drive.google.com/file/d/1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD  ║
║     Save to: ./weights/detector2.pt                                          ║
║                                                                              ║
║  OPTIONAL WEIGHTS (for advanced features):                                   ║
║  ─────────────────────────────────────────                                   ║
║                                                                              ║
║  3. Card Classifier:                                                         ║
║     URL: https://drive.google.com/drive/folders/1Ely1gIOEOui7uHLppeS7tLXNtdkvit07
║     Save to: ./weights/card_classifier/                                      ║
║                                                                              ║
║  4. Elixir Classifier:                                                       ║
║     URL: https://drive.google.com/drive/folders/1cuqD_WQaa4uOlzSVEqLUwGmy0XNucteU
║     Save to: ./weights/elixir_classifier/                                    ║
║                                                                              ║
║  ALTERNATIVE: Use gdown (pip install gdown)                                  ║
║  ───────────────────────────────────────────                                 ║
║                                                                              ║
║    gdown 1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_ -O weights/detector1.pt          ║
║    gdown 1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD -O weights/detector2.pt          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def check_weights():
    """Check which weights are available."""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    print("\nWeight Status:")
    print("-" * 50)
    
    required = [
        ("detector1.pt", "YOLOv8 Detector 1"),
        ("detector2.pt", "YOLOv8 Detector 2"),
    ]
    
    optional = [
        ("card_classifier/", "Card Classifier"),
        ("elixir_classifier/", "Elixir Classifier"),
    ]
    
    all_present = True
    
    for filename, desc in required:
        path = weights_dir / filename
        if path.exists():
            print(f"  ✓ {desc}: {path}")
        else:
            print(f"  ✗ {desc}: MISSING ({path})")
            all_present = False
    
    print()
    for filename, desc in optional:
        path = weights_dir / filename
        if path.exists():
            print(f"  ✓ {desc}: {path}")
        else:
            print(f"  ○ {desc}: Not found (optional)")
    
    return all_present


def setup_katacr_integration():
    """Setup symbolic links or copies from KataCR repo if present."""
    katacr_path = Path("../KataCR")
    
    if not katacr_path.exists():
        print("KataCR repo not found at ../KataCR")
        return False
    
    print(f"\nFound KataCR at: {katacr_path.absolute()}")
    
    # Check for runs directory with weights
    runs_dir = katacr_path / "runs"
    if runs_dir.exists():
        print(f"  Found runs directory: {runs_dir}")
        
        # Look for detector weights
        for item in runs_dir.iterdir():
            if "yolov8" in item.name.lower() or "detector" in item.name.lower():
                print(f"    Found: {item.name}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_weights()
    else:
        print_download_instructions()
        print("\n")
        check_weights()
        print("\n")
        setup_katacr_integration()
