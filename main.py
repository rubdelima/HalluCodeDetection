from __future__ import annotations

def main(args) -> None:
    from src.core import ui
    from src.constants import HalluCodeDetectionConfig
    
    config = HalluCodeDetectionConfig(args.config)

    if args.build_dataset:
        from src.dataset.build import build_dataset
        build_dataset(config.dataset_building_config)

    if args.dataset_judge:
        from src.dataset.augmentation import dataset_judge
        dataset_judge(config.dataset_building_config)

    if args.train_model:
        with ui.console.status("Loading training dependencies..."):
            from src.training import train_models
        train_models(config)

    if args.evaluate:
        with ui.console.status("Loading evaluation dependencies..."):
            from src.evaluations import evaluate_models
        evaluate_models(config)
        return
        
    if any([args.build_dataset, args.dataset_judge, args.train_model, args.evaluate]):
        return

    selection = ui.interactive_menu(
        {
            "1": "build_dataset",
            "2": "dataset_judge",
            "3": "train_model",
            "4": "evaluate",
            "q": "quit",
        }
    )
    
    if selection == "build_dataset":
        from src.dataset.build import build_dataset
        build_dataset(config.dataset_building_config)
    
    elif selection == "dataset_judge":
        from src.dataset.augmentation import dataset_judge
        dataset_judge(config.dataset_building_config)
        
    elif selection == "train_model":
        with ui.console.status("Loading training dependencies..."):
            from src.training import train_models
        train_models(config)
    
    elif selection == "evaluate":
        with ui.console.status("Loading evaluation dependencies..."):
            from src.evaluations import evaluate_models
        evaluate_models(config)
        
    else:
        ui.console.print("Bye.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HalluCodeDetection")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--build_dataset", action="store_true")
    parser.add_argument("--dataset_judge", action="store_true")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    main(parser.parse_args())
