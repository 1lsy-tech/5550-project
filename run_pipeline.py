from src.pipeline import Config, run_all
from pprint import pprint

if __name__ == "__main__":
    cfg = Config()
    results = run_all(cfg)
    print("=== Metrics ===")
    pprint(results)
    print("\nArtifacts saved to:", cfg.outputs_dir)
