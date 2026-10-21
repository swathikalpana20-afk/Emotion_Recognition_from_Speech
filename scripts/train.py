"""
EMOTISENSE AI — Standalone training script.
Usage:
    python scripts/train.py --dataset ravdess [--epochs 50] [--batch-size 32] [--model cnn_bilstm]
"""
import argparse
import sys
import logging
from pathlib import Path

# Add backend to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train EMOTISENSE AI model")
    parser.add_argument("--dataset", default="ravdess", choices=["ravdess", "tess", "emodb"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", default="cnn_bilstm", choices=["cnn_bilstm", "cnn_baseline"])
    args = parser.parse_args()

    from app.services.training import create_job, start_training_job, get_job, STATUS_COMPLETED, STATUS_FAILED
    import time

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "model_type": args.model,
    }

    logger.info(f"Starting training: dataset={args.dataset}, config={config}")
    job_id = create_job(args.dataset, config)
    start_training_job(job_id)

    # Poll until done
    while True:
        job = get_job(job_id)
        logger.info(f"[{job.status}] {job.message}")
        if job.status in (STATUS_COMPLETED, STATUS_FAILED):
            break
        time.sleep(5)

    job = get_job(job_id)
    if job.status == STATUS_COMPLETED:
        logger.info("Training completed successfully!")
        logger.info(f"Final message: {job.message}")
        return 0
    else:
        logger.error(f"Training failed: {job.error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
