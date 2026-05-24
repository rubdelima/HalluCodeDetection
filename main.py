from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.core import ui
from src.dataset.augmentation import dataset_judge
from src.dataset.build import build_dataset
from src.dataset.view_textual import view_dataset


def interactive_menu() -> str:
    options = {
        "1": "build_dataset",
        "2": "dataset_judge",
        "3": "train_model",
        "4": "evaluate",
        "5": "view_dataset",
        "q": "quit",
    }
    ui.console.print("Select an option:")
    
    for key, value in options.items():
        ui.console.print(f"  {key}) {value}")
    
    choice = ui.console.input("Choice: ").strip().lower()
    
    return options.get(choice, "")


def load_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="HalluCodeDetection")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--build_dataset", action="store_true")
    parser.add_argument("--dataset_judge", action="store_true")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--view_dataset", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.build_dataset:
        build_dataset(config)

    if args.dataset_judge:
        dataset_judge(config, model_name=args.model_name)

    if args.train_model:
        ui.console.print("Train model not implemented yet.")

    if args.evaluate:
        ui.console.print("Evaluate not implemented yet.")

    if args.view_dataset:
        view_dataset(config)
        return
        
    if any([args.build_dataset, args.dataset_judge, args.train_model, args.evaluate, args.view_dataset]):
        return

    selection = interactive_menu()
    
    if selection == "build_dataset":
        build_dataset(config)
    elif selection == "dataset_judge":
        dataset_judge(config, model_name=None)
    elif selection == "train_model":
        ui.console.print("Train model not implemented yet.")
    elif selection == "evaluate":
        ui.console.print("Evaluate not implemented yet.")
    elif selection == "view_dataset":
        view_dataset(config)
    else:
        ui.console.print("Bye.")


if __name__ == "__main__":
    main()
