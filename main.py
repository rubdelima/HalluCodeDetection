from __future__ import annotations

import faulthandler
import traceback


def main(args) -> None:
    faulthandler.enable()

    from src.constants import HalluCodeDetectionConfig
    from src.core import ui

    config = HalluCodeDetectionConfig(args.config)

    if args.build_dataset:
        from src.dataset.build import build_dataset
        try:
            build_dataset(config.dataset_building_config)
        except Exception:
            traceback.print_exc()
            raise

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
        evaluate_models(config, retry=args.retry, model_names=args.model_name)
        return

    if args.phase5:
        from src.static_analysis import run_static_analysis
        run_static_analysis(
            config,
            selected_tools=args.phase5_tool,
            retry=args.retry,
            batch_size=args.phase5_batch_size,
            limit=args.phase5_limit,
        )
        return

    if args.phase6:
        from src.phase6 import run_phase6
        run_phase6(
            config,
            retry=args.retry,
            model_names=args.model_name,
            limit=args.phase6_limit,
            max_rounds=args.phase6_rounds,
        )
        return

    if args.phase7:
        from src.phase7 import run_phase7
        run_phase7(
            config,
            retry=args.retry,
            model_names=args.model_name,
            limit=args.phase7_limit,
            only_errors=args.phase7_only_errors,
        )
        return
        
    if any([
        args.build_dataset,
        args.dataset_judge,
        args.train_model,
        args.evaluate,
        args.phase5,
        args.phase6,
        args.phase7,
    ]):
        return

    selection = ui.interactive_menu(
        {
            "1": "build_dataset",
            "2": "dataset_judge",
            "3": "train_model",
            "4": "evaluate",
            "5": "phase5",
            "6": "phase6",
            "7": "phase7",
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

    elif selection == "phase5":
        from src.static_analysis import run_static_analysis
        run_static_analysis(config)

    elif selection == "phase6":
        from src.phase6 import run_phase6
        run_phase6(config)

    elif selection == "phase7":
        from src.phase7 import run_phase7
        run_phase7(config)
        
    else:
        ui.console.print("Bye.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HalluCodeDetection")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--build_dataset", action="store_true")
    parser.add_argument("--dataset_judge", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        action="append",
        default=None,
        help="Select a configured model for Phase 4, 6, or 7; repeat for multiple models.",
    )
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--phase5",
        "--static_analysis",
        action="store_true",
        help="Compare static-analysis tools on runtime and syntax recall.",
    )
    parser.add_argument(
        "--phase5_tool",
        action="append",
        choices=("compile", "ruff", "pyright", "pylint", "semgrep", "crosshair"),
        default=None,
        help="Run only this Phase-5 tool; repeat to select multiple tools.",
    )
    parser.add_argument(
        "--phase5_batch_size",
        type=int,
        default=250,
        help="Number of samples committed per Phase-5 batch (default: 250).",
    )
    parser.add_argument(
        "--phase5_limit",
        type=int,
        default=None,
        help="Limit samples for a smoke test; omit for the full dataset.",
    )
    parser.add_argument(
        "--phase6",
        action="store_true",
        help="Generate and self-review code with compile + Pyright feedback.",
    )
    parser.add_argument(
        "--phase6_limit",
        type=int,
        default=None,
        help="Limit Phase-6 programming tasks per model for a smoke test.",
    )
    parser.add_argument(
        "--phase6_rounds",
        type=int,
        default=None,
        help="Override the configured maximum number of Phase-6 rounds.",
    )
    parser.add_argument(
        "--phase7",
        action="store_true",
        help="Classify held-out Phase-1 code with compile + Pyright feedback.",
    )
    parser.add_argument(
        "--phase7_limit",
        type=int,
        default=None,
        help="Limit Phase-7 Phase-1 samples per model for a smoke test.",
    )
    parser.add_argument(
        "--phase7_only_errors",
        action="store_true",
        help="Select only Phase-1 rows whose ground truth is not correct.",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry failed resumable samples instead of skipping them.",
    )
    main(parser.parse_args())
