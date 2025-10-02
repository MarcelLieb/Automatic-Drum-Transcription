import numpy as np

from main import main as train
from settings import Config


def main():
    config = Config()
    scores = []
    for fold in range(config.dataset.k_folds):
        config.dataset.fold = fold
        _, score = train(
            config=config, metric_to_track="F-Score/Sum/Validation", seed=42
        )
        scores.append(score)
    print(f"Scores for folds {config.dataset.k_folds}: {scores}")
    print(f"Stats: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
