import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WEIGHTS = PROJECT_ROOT / 'models' / 'best_model.pth'
TARGET_WEIGHTS = PROJECT_ROOT / 'models' / 'final_model.pth'


def main():
    if not SOURCE_WEIGHTS.exists():
        raise FileNotFoundError(f'Source weights not found: {SOURCE_WEIGHTS}')
    TARGET_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_WEIGHTS, TARGET_WEIGHTS)
    print(f'Final model weights saved to: {TARGET_WEIGHTS}')


if __name__ == '__main__':
    main()
