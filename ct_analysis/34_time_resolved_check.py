"""How much time-resolved (multi-phase) cardiac CT is actually in the SCAPIS data.

Reads the series sheets that 04_build_site_dicom_inventories.py wrote and counts,
per patient, how many distinct cardiac phases were reconstructed. A patient with
one phase is an ordinary single-phase CCTA; two phases is the pair Lee has seen;
five or more starts to be a usable cardiac cycle.

Phase evidence, in order of trust:
    1. NominalPercentageOfCardiacPhase in the header (cardiac_phase_percentages)
    2. a percentage in the series description, e.g. "75% best diast"
    3. the scan options field, e.g. TP075PC
    4. several temporal positions inside one series (a true 4D series)
Script 04 already merges 1 to 3 into phase_for_family and flags 4 in volume_class.

Usage
    python 34_time_resolved_check.py --folder "Q:\\users\\marfi\\CT_analysis"
    python 34_time_resolved_check.py --inventory "Q:\\...\\all_series_inventory_6sites.csv"

Output
    time_resolved\\time_resolved.xlsx   by site, by patient, and the phase histogram
    time_resolved\\phases_per_patient.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PERCENT = re.compile(r"(?<!\d)(\d{1,3})\s*%")
SCAN_OPTION = re.compile(r"(?:TP|BESTPH_D_P)0*(\d{2,3})PC", re.IGNORECASE)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=Path("."),
                        help="where the inventory workbooks are")
    parser.add_argument("--inventory", default="",
                        help="one file instead of the folder: .xlsx, .csv or .csv.gz")
    parser.add_argument("--pattern", default="site_*_dicom_inventory.xlsx")
    parser.add_argument("--sheet", default="Series")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cycle", type=int, default=5,
                        help="phases from here up are called a cardiac cycle")
    return parser.parse_known_args(argv)[0]


def phases_in(text: object) -> set:
    """Every cardiac phase percentage mentioned in one field."""
    value = str(text or "").strip()
    if not value or value.lower() in {"nan", "none"}:
        return set()
    found = set()
    for part in re.split(r"[,;|]", value):
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(r"\d{1,3}(\.\d+)?", part):
            found.add(round(float(part)))
            continue
        for match in PERCENT.finditer(part):
            found.add(int(match.group(1)))
        for match in SCAN_OPTION.finditer(part):
            found.add(int(match.group(1)))
    return {value for value in found if 0 <= value <= 100}


def read_series(args: argparse.Namespace) -> pd.DataFrame:
    paths = ([Path(args.inventory)] if args.inventory
             else sorted(args.folder.glob(args.pattern)))
    if not paths:
        raise SystemExit(f"no inventory found: {args.folder / args.pattern}")
    frames = []
    for path in paths:
        if path.suffix.lower() in {".csv", ".gz", ".txt"}:
            table = pd.read_csv(path, low_memory=False)
        else:
            sheets = pd.read_excel(path, sheet_name=None, nrows=0)
            name = args.sheet if args.sheet in sheets else list(sheets)[0]
            table = pd.read_excel(path, sheet_name=name)
        table.columns = [str(column).strip() for column in table.columns]
        print(f"  {path.name}: {len(table)} series")
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def per_patient(series: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """One row per patient: how many phases, and whether a cycle was reconstructed."""
    columns = {name.lower(): name for name in series.columns}
    def column(*names, default=""):
        for name in names:
            if name in columns:
                return series[columns[name]]
        return pd.Series([default] * len(series), index=series.index)

    frame = pd.DataFrame({
        "site": column("site_number", "site").astype(str),
        "patient_id": column("patient_id").astype(str),
        "volume_class": column("volume_class").astype(str),
        "modality": column("modality").astype(str),
        "protocol": column("protocol", "ccta_protocol", "scan_options").astype(str),
        "description": column("series_description").astype(str),
    })
    text = (column("cardiac_phase_percentages").astype(str) + "|"
            + column("phase_for_family").astype(str) + "|"
            + column("series_description").astype(str) + "|"
            + column("scan_options").astype(str))
    frame["phases"] = [phases_in(value) for value in text]
    inside = pd.to_numeric(column("n_temporal_positions", "temporal_positions_estimated",
                                  "n_trigger_times", default=0), errors="coerce").fillna(0)
    frame["inside_series"] = inside.astype(int)

    rows = []
    for (site, patient), group in frame.groupby(["site", "patient_id"], dropna=False):
        phases = set().union(*group["phases"]) if len(group) else set()
        within = int(group["inside_series"].max())
        total = max(len(phases), within)
        rows.append({
            "site": site,
            "patient_id": patient,
            "series": len(group),
            "distinct_phases": len(phases),
            "max_phases_in_one_series": within,
            "phases": ", ".join(str(value) for value in sorted(phases)),
            "time_resolved": total >= 2,
            "cycle": total >= cycle,
            "four_d_series": int(group["volume_class"].str.startswith("4D").sum()),
            "possible_four_d": int((group["volume_class"] == "possible_4D").sum()),
        })
    return pd.DataFrame(rows)


def by_site(patients: pd.DataFrame, cycle: int) -> pd.DataFrame:
    table = patients.groupby("site").agg(
        patients=("patient_id", "nunique"),
        series=("series", "sum"),
        one_phase=("distinct_phases", lambda values: int((values <= 1).sum())),
        two_phases=("distinct_phases", lambda values: int((values == 2).sum())),
        three_or_four=("distinct_phases", lambda values: int(values.between(3, 4).sum())),
        cycle=("cycle", "sum"),
        time_resolved=("time_resolved", "sum"),
        true_4d_series=("four_d_series", "sum"),
    ).reset_index()
    table[f"cycle ({cycle}+ phases) %"] = (100 * table["cycle"] / table["patients"]).round(1)
    table["time resolved %"] = (100 * table["time_resolved"] / table["patients"]).round(1)
    return table


def main(argv=None) -> None:
    args = parse_args(argv)
    folder = (args.output or (args.folder / "time_resolved"))
    folder.mkdir(parents=True, exist_ok=True)

    print("1. reading the inventories")
    series = read_series(args)
    print(f"   {len(series)} series in total")

    print("2. counting the cardiac phases per patient")
    patients = per_patient(series, args.cycle)
    sites = by_site(patients, args.cycle)
    histogram = (patients["distinct_phases"].clip(upper=20)
                 .value_counts().sort_index().rename("patients").reset_index())
    histogram.columns = ["distinct_phases", "patients"]

    total = len(patients)
    print(f"\n   {total} patients")
    print(f"   time resolved (2 or more phases) {int(patients['time_resolved'].sum())}"
          f" ({100 * patients['time_resolved'].mean():.1f} %)")
    print(f"   a cardiac cycle ({args.cycle}+ phases)  {int(patients['cycle'].sum())}"
          f" ({100 * patients['cycle'].mean():.1f} %)")
    print(f"   true 4D series (several phases in one series) "
          f"{int(patients['four_d_series'].sum())}")
    print("\n   phases per patient:")
    for _, row in histogram.iterrows():
        print(f"     {int(row['distinct_phases']):>3} phases  {int(row['patients']):>6} patients")

    book = folder / "time_resolved.xlsx"
    with pd.ExcelWriter(book) as writer:
        sites.to_excel(writer, sheet_name="By_site", index=False)
        histogram.to_excel(writer, sheet_name="Phases_per_patient", index=False)
        patients.sort_values(["distinct_phases", "site"], ascending=[False, True]).to_excel(
            writer, sheet_name="Patients", index=False)
        patients[patients["time_resolved"]].to_excel(
            writer, sheet_name="Time_resolved_only", index=False)

    plt.figure(figsize=(7.2, 4.2))
    plt.bar(histogram["distinct_phases"], histogram["patients"], color="#4c78a8")
    plt.xlabel("distinct cardiac phases reconstructed")
    plt.ylabel("patients")
    plt.title("How time resolved is the SCAPIS cardiac CT")
    plt.tight_layout()
    plt.savefig(folder / "phases_per_patient.png", dpi=200)
    plt.close()
    print(f"\nExcel : {book}")
    print(f"Plot  : {folder / 'phases_per_patient.png'}")


if __name__ == "__main__":
    main()
