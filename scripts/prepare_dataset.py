"""
EMOTISENSE AI — Dataset preparation and validation script.
Usage:
    python scripts/prepare_dataset.py --dataset ravdess
"""
import argparse
import sys
import logging
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate and summarize a dataset")
    parser.add_argument("--dataset", default="ravdess", choices=["ravdess", "tess", "emodb"])
    args = parser.parse_args()

    from app.ml.datasets import dataset_summary
    from app.core.config import DATA_DIR

    dataset_dir = DATA_DIR / args.dataset
    logger.info(f"Scanning: {dataset_dir}")

    summary = dataset_summary(args.dataset, dataset_dir)

    print("\n" + "=" * 50)
    print(f"Dataset: {summary['dataset_name'].upper()}")
    print(f"Available: {summary['available']}")
    print(f"Total samples: {summary.get('total_samples', 0)}")
    print(f"Speakers: {summary.get('num_speakers', 'N/A')}")
    print("\nClass distribution:")
    for label, count in sorted(summary.get("class_counts", {}).items()):
        bar = "█" * (count // 5)
        print(f"  {label:12s}: {count:4d} {bar}")
    if summary.get("missing_classes"):
        print(f"\nMissing classes: {summary['missing_classes']}")
    print(f"\nMessage: {summary['message']}")
    print("=" * 50)

    if not summary["available"]:
        print("\n⚠  Dataset not found. Please download it and place at:")
        print(f"   {dataset_dir}")
        print("\n  RAVDESS: https://zenodo.org/record/1188976")
        print("  TESS:    https://tspace.library.utoronto.ca/handle/1807/24487")
        print("  EMO-DB:  http://www.emodb.org/")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
