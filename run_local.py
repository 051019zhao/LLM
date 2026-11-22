"""Run the competition notebook with the current Python interpreter."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = {
    2: ROOT / "notebooks" / "03_optimized_submission.ipynb",
    3: ROOT / "notebooks" / "04_topic_robust_submission.ipynb",
    4: ROOT / "notebooks" / "05_family_style_experiment.ipynb",
    7: ROOT / "notebooks" / "07_controlled_nbsvm_submission.ipynb",
    8: ROOT / "notebooks" / "08_external_data_submission.ipynb",
}
NOTEBOOKS = {number: path for number, path in NOTEBOOKS.items() if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(os.environ.get("NAME_THAT_LLM_DATA", ROOT / "data")),
        help="Folder containing train.csv and test.csv (default: repository data folder)",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "local_results")
    parser.add_argument(
        "--extra-file", type=Path,
        default=Path(os.environ.get("NAME_THAT_LLM_EXTRA", ROOT / "data/external/crtproduction__name-that-llm-open-pool__v18/ntl_open_pool.parquet")),
        help="Pinned Open Pool v18 parquet for round 8; independent of --data-dir",
    )
    parser.add_argument("--round", type=int, choices=NOTEBOOKS, default=2, help="Model experiment round")
    parser.add_argument("--validate", action="store_true", help="Run the selected round's validation")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    for filename in ("train.csv", "test.csv"):
        if not (data_dir / filename).is_file():
            parser.error(f"Missing {data_dir / filename}. Set --data-dir to your competition data folder.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NAME_THAT_LLM_DATA"] = str(data_dir)
    os.environ["NAME_THAT_LLM_VALIDATE"] = "1" if args.validate else "0"
    if args.round == 8:
        if not args.extra_file.is_file():
            parser.error("Round 8 requires Open Pool v18. Run python experiments/download_public_data.py or set --extra-file.")
        os.environ["NAME_THAT_LLM_EXTRA"] = str(args.extra_file.resolve())
    os.chdir(output_dir)

    notebook_path = NOTEBOOKS[args.round]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"display": lambda frame: print(frame.to_string(index=False))}
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{notebook_path.name}:cell {index}", "exec"), namespace)


if __name__ == "__main__":
    main()
