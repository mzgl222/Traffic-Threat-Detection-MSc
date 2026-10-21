import subprocess
import sys
from pathlib import Path

# Główny katalog projektu
ROOT = Path(__file__).resolve().parent.parent

# Katalog ze skryptami
SCRIPTS = ROOT / "scripts"

skrypty = ["analyze_tracks.py", 
           "analyze_risk_events.py", 
           "analyze_risk_timeline.py", 
           "vqa2csv.py", 
           "overlay_risk_phases.py", 
           "build_frame_dataset.py"
           ]

for skrypt in skrypty:
    print(f"Uruchamiam {skrypt}", flush=True)

    subprocess.run(
        [sys.executable, str(SCRIPTS / skrypt)],
        cwd=ROOT,
        check=True
    )

print("Wszystkie skrypty zakończone!")