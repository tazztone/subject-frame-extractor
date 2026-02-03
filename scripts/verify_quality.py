import json
import sys
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
import statistics

class QualityVerifier:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.mask_meta_path = output_dir / "mask_metadata.json"
        self.db_path = output_dir / "metadata.db"
        self.scenes_path = output_dir / "scenes.json"
        
        self.report = {
            "status": "PASS",
            "warnings": [],
            "errors": [],
            "metrics": {}
        }

    def verify(self) -> Dict[str, Any]:
        print(f"🔍 Analyzing results in {self.output_dir}...")
        
        if not self.mask_meta_path.exists():
            self._fail("mask_metadata.json missing")
            return self.report

        if not self.db_path.exists():
            self._fail("metadata.db missing")
            return self.report

        self._check_mask_quality()
        self._check_db_metrics()
        
        return self.report

    def _fail(self, msg: str):
        self.report["status"] = "FAIL"
        self.report["errors"].append(msg)
        print(f"❌ {msg}")

    def _warn(self, msg: str):
        self.report["warnings"].append(msg)
        print(f"⚠️ {msg}")

    def _check_mask_quality(self):
        try:
            with open(self.mask_meta_path) as f:
                data = json.load(f)
            
            total_frames = len(data)
            if total_frames == 0:
                self._fail("No frames in mask metadata")
                return

            valid_masks = sum(1 for v in data.values() if not v.get("mask_empty", True))
            yield_rate = (valid_masks / total_frames) * 100
            
            self.report["metrics"]["mask_yield"] = f"{yield_rate:.1f}%"
            print(f"📊 Mask Yield: {yield_rate:.1f}% ({valid_masks}/{total_frames})")

            # Threshold: Fail if < 50% yield (adjustable)
            if yield_rate < 50:
                self._fail(f"Mask yield too low ({yield_rate:.1f}% < 50%)")
            
            # Check for consecutive failures (Tracking Loss)
            frames = sorted(data.keys())
            consecutive_failures = 0
            max_consecutive = 0
            
            for frame in frames:
                if data[frame].get("mask_empty", True):
                    consecutive_failures += 1
                else:
                    max_consecutive = max(max_consecutive, consecutive_failures)
                    consecutive_failures = 0
            
            max_consecutive = max(max_consecutive, consecutive_failures)
            self.report["metrics"]["max_consecutive_drops"] = max_consecutive
            
            if max_consecutive >= 5:
                self._warn(f"Tracking lost for {max_consecutive} consecutive frames")

        except Exception as e:
            self._fail(f"Failed to parse mask metadata: {str(e)}")

    def _check_db_metrics(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get metrics (JSON string) and face_sim
            cursor.execute("SELECT metrics, face_sim FROM metadata")
            rows = cursor.fetchall()
            
            if not rows:
                self._fail("Database is empty")
                conn.close()
                return

            niqe_scores = []
            face_sims = []
            
            for metrics_json, face_sim in rows:
                if face_sim is not None:
                    face_sims.append(face_sim)
                
                if metrics_json:
                    try:
                        metrics = json.loads(metrics_json)
                        # Check for niqe_score or quality_score
                        score = metrics.get("niqe_score") or metrics.get("quality_score")
                        if score:
                            niqe_scores.append(float(score))
                    except json.JSONDecodeError:
                        pass
            
            if not niqe_scores:
                self._fail("All NIQE/Quality scores are missing or zero")
            else:
                avg_niqe = statistics.mean(niqe_scores)
                self.report["metrics"]["avg_niqe"] = round(avg_niqe, 2)
                print(f"📊 Avg NIQE: {avg_niqe:.2f} (Lower is better)")
            
            if face_sims:
                avg_sim = statistics.mean(face_sims)
                self.report["metrics"]["avg_face_sim"] = round(avg_sim, 2)
                print(f"📊 Avg Face Sim: {avg_sim:.2f}")

            conn.close()

        except Exception as e:
            self._fail(f"Database check failed: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_quality.py <output_dir>")
        sys.exit(1)
        
    verifier = QualityVerifier(Path(sys.argv[1]))
    report = verifier.verify()
    
    if report["status"] == "FAIL":
        sys.exit(1)
    sys.exit(0)
