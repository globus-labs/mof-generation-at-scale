"""
This extraction step can take roughly 10-20 minutes, depending on the machine.
Specifically, it ran for roughly 13 minutes on an M1 Max Apple Silicon machine.
"""

import gzip
import json
import matplotlib.pyplot as plt
import pandas as pd
import typing as t

from datetime import datetime
from pathlib import Path
from time import perf_counter
from tqdm import tqdm

FIG_DIR: t.Final[Path] = Path("figures")
FILENAME: t.Final[str] = "mofs.json.gz"
RUN_DIR: t.Final[Path] = Path("../prod-runs/")
SUM_DIR: t.Final[Path] = Path("summaries")


def extract_mofs(directory: Path) -> None:
    """Extract the MOFs from a workflow run.

    Specifically, this function looks at the given directory and looks for the
    file named "mofs.json.gz" (see `FILENAME`) in the directory. It then extracts
    the relevant features into a DataFrame and produces a summarized dataset
    that gets dropped in a summaries dataset (see `SUM_DIR`).

    Args:
        directory (Path): Directory where the outputted MOFs can be found from a
            given run.
    """
    records = []

    with gzip.open(directory / FILENAME, "rt") as fp:
        for line in fp:
            record = json.loads(line)

            # Remove structure data, label linkers by anchor
            for k in ["md_trajectory", "nodes", "structure", "_id"]:
                del record[k]

            for ligand in record.pop("ligands"):
                record[f'ligand.{ligand["anchor_type"]}'] = ligand
                for k in ["xyz", "dummy_element", "anchor_type"]:
                    del ligand[k]

            record["time"] = record.pop("times")["created"]["$date"]
            records.append(pd.json_normalize(record))

    # Convert to `DataFrame` and format features.
    records = pd.concat(records, ignore_index=True)
    records["model_version"] = records[
        ["ligand.cyano.metadata.model_version", "ligand.COO.metadata.model_version"]
    ].max(axis=1)
    records["time"] = records["time"].apply(
        lambda x: datetime.strptime(
            x[: x.index(".")] + "Z" if "." in x else x, "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    records["cumulative_found"] = (records["structure_stability.uff"] < 0.1).cumsum()
    records["walltime"] = (records["time"] - records["time"].min()).apply(
        lambda x: x.total_seconds()
    )
    records.sort_values("walltime", inplace=True)

    # Save the result summaries.
    SUM_DIR.mkdir(exist_ok=True)
    path = SUM_DIR / f"{directory.name}.csv.gz"
    records.to_csv(path, index=False)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)

    start_time = perf_counter()
    dirs = list(RUN_DIR.glob("**/mofs.json.gz"))
    pbar = tqdm(total=len(dirs))
    for d in dirs:
        pbar.set_description("Processing `{}`".format(d.parent))
        extract_mofs(d.parent)
        pbar.update()

    print(f"Time taken (sec.): {perf_counter() - start_time}")
