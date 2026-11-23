from Photon_Simulator.utils.logger import setup_logger,get_logger
from Photon_Simulator.config.settings import settings
import pandas as pd 
import numpy as np
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonStatus




def generate_isotropic_photons(point, N, status):
    """
    Generate N photons emitted isotropically from a given point.

    Parameters
    ----------
    point : array-like (3,)
        Interaction point in 3D space.
    N : int
        Number of photons (and DataFrames).
    status : PhotonStatus
        Initial photon status.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames, one per photon.
    """

    point = np.asarray(point)

    # Sample isotropic directions
    phi = np.random.uniform(0, 2*np.pi, N)
    cos_theta = np.random.uniform(-1, 1, N)
    sin_theta = np.sqrt(1 - cos_theta**2)

    directions = np.stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ], axis=1)

    # Build list of DataFrames
    dfs = [
        pd.DataFrame({
            "interaction_point": [point.copy()],
            "direction": [directions[i]],
            "status": [status],
            "time": 0.0
        })
        for i in range(N)
    ]

    return dfs

def multiple_starting_point(
    N, N_photon, Max_x, Max_y, Max_z, max_workers=None,
    photon_status=PhotonStatus.RUNNING_INSIDE, chunksize=100,
    random_seed=None, verbose=True, *, scene=None, engine="approximate",
    chunk_size=1000, max_steps=20000, return_tracks=False,
):
    """Forward transport in cm/ns. Legacy worker arguments remain accepted.

    chunk_size schedules JAX batches in this process. Histories are stored
    only when return_tracks=True, in the ordered trajectory column.
    """
    from Photon_Simulator.core.detectorSetup.referenceScene import build_reference_scene
    from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch
    from Photon_Simulator.core.photonsEngines.batchTransport import BatchTransport
    rng = np.random.default_rng(random_seed)
    seed = int(rng.integers(0, 2**32)) if random_seed is None else random_seed
    points = np.repeat(rng.uniform([0, 0, 0], [Max_x, Max_y, Max_z], size=(N, 3)), N_photon, axis=0)
    z = rng.uniform(-1, 1, len(points))
    phi = rng.uniform(0, 2*np.pi, len(points))
    radial = np.sqrt(1-z*z)
    directions = np.column_stack((radial*np.cos(phi), radial*np.sin(phi), z))
    batch = PhotonBatch(points, directions, np.zeros(len(points)),
                        np.full(len(points), int(photon_status), dtype=np.int32))
    runner = BatchTransport(scene or build_reference_scene(), engine=engine,
                            seed=seed, chunk_size=chunk_size, max_steps=max_steps)
    result = runner.run(batch, return_tracks=return_tracks)
    records = pd.DataFrame({
        "origin_index": np.repeat(np.arange(N), N_photon),
        "starting_point": list(points), "starting_direction": list(directions),
        "arrival_time": result.batch.times,
        "arrival_status": [PhotonStatus(int(s)) for s in result.batch.statuses],
        "n_interactions": result.steps, "truncated": result.truncated,
    })
    if return_tracks:
        records["trajectory"] = result.tracks
    records.attrs["transport"] = runner.device_info
    if verbose:
        print(runner.device_info)
        print(f"Generated: {len(records)}; truncated: {int(result.truncated.sum())}")
    return records


#! THANKS TO AI
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def _expand_vector_column(df, column_name, prefix):
    """
    Expand a column containing 3D vectors/lists/arrays into three scalar columns.

    Example:
    starting_point -> starting_point_x, starting_point_y, starting_point_z
    """
    vectors = df[column_name].apply(
        lambda v: np.asarray(v, dtype=float) if v is not None else np.array([np.nan, np.nan, np.nan], dtype=float)
    )

    expanded = pd.DataFrame(vectors.tolist(), columns=[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"], index=df.index)
    return expanded


def show_photon_dashboards(final_df, bins=30, figsize=(16, 10)):
    """
    Show 4 dashboards from the photon simulation output DataFrame.

    Required columns in final_df:
    - origin_index
    - starting_point
    - starting_direction
    - arrival_time
    - arrival_status
    - n_interactions
    """
    df = final_df.copy()

    # Convert status to string for plotting/grouping
    df["arrival_status"] = df["arrival_status"].map(_status_label)
    if "truncated" in df:
        df.loc[df["truncated"].fillna(False), "arrival_status"] = "TRUNCATED"

    # Expand vector columns
    sp = _expand_vector_column(df, "starting_point", "starting_point")
    sd = _expand_vector_column(df, "starting_direction", "starting_direction")
    df = pd.concat([df, sp, sd], axis=1)

    point_cols = ["starting_point_x", "starting_point_y", "starting_point_z"]
    dir_cols = ["starting_direction_x", "starting_direction_y", "starting_direction_z"]

    # Unique origin indices and statuses
    origin_values = sorted(df["origin_index"].dropna().unique())
    status_values = sorted(df["arrival_status"].dropna().unique())

    # ==========================================================
    # Dashboard 1: arrival analysis
    # ==========================================================
    fig1, axes1 = plt.subplots(2, 3, figsize=figsize)
    fig1.suptitle("Dashboard 1: arrival analysis", fontsize=16)

    # Row 1: starting point coordinate vs n_interactions
    for j, col in enumerate(point_cols):
        ax = axes1[0, j]
        for origin_idx in origin_values:
            mask = df["origin_index"] == origin_idx
            ax.scatter(
                df.loc[mask, col],
                df.loc[mask, "n_interactions"],
                label=f"origin {origin_idx}",
                alpha=0.7
            )
        ax.set_title(f"{col} vs n_interactions")
        ax.set_xlabel(col)
        ax.set_ylabel("n_interactions")

    # Row 2: starting direction coordinate vs arrival_time
    for j, col in enumerate(dir_cols):
        ax = axes1[1, j]
        for origin_idx in origin_values:
            mask = df["origin_index"] == origin_idx
            ax.scatter(
                df.loc[mask, col],
                df.loc[mask, "arrival_time"],
                label=f"origin {origin_idx}",
                alpha=0.7
            )
        ax.set_title(f"{col} vs arrival_time")
        ax.set_xlabel(col)
        ax.set_ylabel("arrival_time")

    # Put legend only once to avoid clutter
    if len(origin_values) > 0:
        axes1[0, 0].legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.show()

    # ==========================================================
    # Dashboard 2: status analysis
    # ==========================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=figsize)
    fig2.suptitle("Dashboard 2: status analysis", fontsize=16)

    # Row 1: histograms of starting point coordinates grouped by arrival_status
    for j, col in enumerate(point_cols):
        ax = axes2[0, j]
        data_by_status = [df.loc[df["arrival_status"] == status, col].dropna().values for status in status_values]
        non_empty_data = [arr for arr in data_by_status if len(arr) > 0]
        non_empty_labels = [status for status, arr in zip(status_values, data_by_status) if len(arr) > 0]

        if len(non_empty_data) > 0:
            ax.hist(non_empty_data, bins=bins, label=non_empty_labels, alpha=0.7)
        ax.set_title(f"arrival_status by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    # Row 2: histograms of starting direction coordinates grouped by arrival_status
    for j, col in enumerate(dir_cols):
        ax = axes2[1, j]
        data_by_status = [df.loc[df["arrival_status"] == status, col].dropna().values for status in status_values]
        non_empty_data = [arr for arr in data_by_status if len(arr) > 0]
        non_empty_labels = [status for status, arr in zip(status_values, data_by_status) if len(arr) > 0]

        if len(non_empty_data) > 0:
            ax.hist(non_empty_data, bins=bins, label=non_empty_labels, alpha=0.7)
        ax.set_title(f"arrival_status by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # ==========================================================
    # Dashboard 3: Caught photons times
    # ==========================================================
    catched_df = df[df["arrival_status"].str.contains("CAUGHT", case=False, na=False)].copy()

    if catched_df.empty:
        print("Dashboard 3: no Caught photons found.")
    else:
        unique_origins = sorted(catched_df["origin_index"].dropna().unique())
        n_origins = len(unique_origins)

        ncols = min(3, n_origins)
        nrows = math.ceil(n_origins / ncols)

        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig3.suptitle("Dashboard 3: Caught photons times", fontsize=16)

        if nrows == 1 and ncols == 1:
            axes3 = np.array([[axes3]])
        elif nrows == 1:
            axes3 = np.array([axes3])
        elif ncols == 1:
            axes3 = np.array([[ax] for ax in axes3])

        flat_axes = axes3.flatten()

        for ax, origin_idx in zip(flat_axes, unique_origins):
            subset = catched_df.loc[catched_df["origin_index"] == origin_idx, "arrival_time"].dropna()
            ax.hist(subset, bins=bins, alpha=0.8)
            ax.set_title(f"origin_index = {origin_idx}")
            ax.set_xlabel("arrival_time")
            ax.set_ylabel("count")

        # Hide unused subplots
        for ax in flat_axes[len(unique_origins):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # ==========================================================
    # Dashboard 4: pie
    # ==========================================================
    status_counts = df["arrival_status"].value_counts(dropna=False)

    fig4, ax4 = plt.subplots(figsize=(8, 8))
    fig4.suptitle('Dashboard 4: pie', fontsize=16)

    ax4.pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax4.set_title("arrival_status distribution")
    plt.tight_layout()
    plt.show()






def dashboard_starting_point_and_direction_vs_time(df):
    """
    Show analysis dashboards for:
    1. each starting-point coordinate vs arrival time
    2. each starting-direction coordinate vs arrival time

    Expected columns in df:
    - 'starting point'      -> [x, y, z]
    - 'starting direction'  -> list of [dx, dy, dz] for the same photons
    - 'time distribution'   -> list of arrival times for the same photons

    Notes
    -----
    Each row of df represents one starting point.
    The lists in 'starting direction' and 'time distribution' should refer to
    the same photons.
    """

    point_records = []
    direction_records = []

    skipped_rows = 0
    truncated_rows = 0
    total_rows = 0

    for _, row in df.iterrows():
        total_rows += 1

        starting_point = row.get("starting point", None)
        directions = row.get("starting direction", None)
        times = row.get("time distribution", None)

        if starting_point is None or directions is None or times is None:
            skipped_rows += 1
            continue

        if len(starting_point) != 3:
            skipped_rows += 1
            continue

        if len(directions) == 0 or len(times) == 0:
            skipped_rows += 1
            continue

        # If lengths do not match, keep only the common part
        n = min(len(directions), len(times))
        if n == 0:
            skipped_rows += 1
            continue

        if len(directions) != len(times):
            truncated_rows += 1

        x0, y0, z0 = starting_point

        for direction, t in zip(directions[:n], times[:n]):
            if direction is None or len(direction) != 3:
                continue

            dx, dy, dz = direction

            point_records.append({
                "x0": x0,
                "y0": y0,
                "z0": z0,
                "time": t
            })

            direction_records.append({
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "time": t
            })

    point_df = pd.DataFrame(point_records)
    direction_df = pd.DataFrame(direction_records)

    if point_df.empty or direction_df.empty:
        print("No valid data available for plotting.")
        print(f"Total rows: {total_rows}")
        print(f"Skipped rows: {skipped_rows}")
        print(f"Rows with mismatched lengths: {truncated_rows}")
        return

    if truncated_rows > 0:
        print(
            f"Warning: {truncated_rows} rows had different lengths between "
            f"'starting direction' and 'time distribution'. "
            f"Only the common part was used."
        )

    # -------- Dashboard 1: starting point coordinates vs arrival time --------
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    axes1[0].scatter(point_df["x0"], point_df["time"], alpha=0.6)
    axes1[0].set_title("Starting x vs arrival time")
    axes1[0].set_xlabel("x0")
    axes1[0].set_ylabel("Arrival time")
    axes1[0].grid(True)

    axes1[1].scatter(point_df["y0"], point_df["time"], alpha=0.6)
    axes1[1].set_title("Starting y vs arrival time")
    axes1[1].set_xlabel("y0")
    axes1[1].set_ylabel("Arrival time")
    axes1[1].grid(True)

    axes1[2].scatter(point_df["z0"], point_df["time"], alpha=0.6)
    axes1[2].set_title("Starting z vs arrival time")
    axes1[2].set_xlabel("z0")
    axes1[2].set_ylabel("Arrival time")
    axes1[2].grid(True)

    fig1.tight_layout()

    # -------- Dashboard 2: starting direction coordinates vs arrival time --------
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    axes2[0].scatter(direction_df["dx"], direction_df["time"], alpha=0.6)
    axes2[0].set_title("Direction x vs arrival time")
    axes2[0].set_xlabel("dx")
    axes2[0].set_ylabel("Arrival time")
    axes2[0].grid(True)

    axes2[1].scatter(direction_df["dy"], direction_df["time"], alpha=0.6)
    axes2[1].set_title("Direction y vs arrival time")
    axes2[1].set_xlabel("dy")
    axes2[1].set_ylabel("Arrival time")
    axes2[1].grid(True)

    axes2[2].scatter(direction_df["dz"], direction_df["time"], alpha=0.6)
    axes2[2].set_title("Direction z vs arrival time")
    axes2[2].set_xlabel("dz")
    axes2[2].set_ylabel("Arrival time")
    axes2[2].grid(True)

    fig2.tight_layout()
    plt.show()











def _status_label(value):
    try:
        return PhotonStatus(int(value)).name
    except (ValueError, TypeError):
        return str(value)


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Photon forward transport (cm, ns)")
    parser.add_argument("--engine", choices=["approximate", "jax"], default="approximate")
    parser.add_argument("--photons", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--return-tracks", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--output", default="photon_results.pkl")
    args = parser.parse_args()
    if min(args.photons, args.max_steps, args.chunk_size) <= 0:
        parser.error("photons, max-steps and chunk-size must be positive")
    setup_logger(settings.logger_name, settings.log_level)
    result = multiple_starting_point(
        args.photons, 1, 1.0, 3.0, 0.5, random_seed=args.seed,
        engine=args.engine, max_steps=args.max_steps, chunk_size=args.chunk_size,
        return_tracks=args.return_tracks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(output)
    print(result["arrival_status"].map(_status_label).value_counts())
    print(f"Saved {output}")
    if args.dashboard:
        show_photon_dashboards(result)


if __name__ == "__main__":
    main()
