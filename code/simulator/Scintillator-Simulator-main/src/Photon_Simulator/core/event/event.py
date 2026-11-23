import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Sequence
import h5py
import numpy as np
import pandas as pd
from rich.progress import MofNCompleteColumn, Progress, BarColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

from Photon_Simulator.core.detectorSetup.scene import Scene
from Photon_Simulator.core.photonsEngines.approximateEngine import (
    PhotonEvent,
    ApproximateEngine,
    PhotonStatus,
    PhotonBatch,
    PhotonBatchSimulator,
    ParallelConfig
)

from Photon_Simulator.utils.logger import get_logger
from Photon_Simulator.config.settings import Settings
from Photon_Simulator.core.physics.particles import MIP, Particle
from Photon_Simulator.core.physics.physics import ParticleInteraction
from Photon_Simulator.config.defaults import DX, DE_DX, N_CORE, CHUNK_SIZE 
from pathlib import Path
import uproot





class Event:
    def __init__(self, scene:Scene=None, particle:Particle = MIP(), logger=None):
        self.log = get_logger(Settings.logger_name) if logger is None else logger
        if scene is None:
            raise ValueError("Event: scene cannot be NoneType")
        self.scene:Scene = scene
        self.particle:Particle = particle
        
        
    # =================================================
    #             EVENT SIMULATION ROUTINE
    # =================================================
    def real_sim(self, x0, direction, length, beta):
        rng = np.random.default_rng(72)
        PI = ParticleInteraction(self.scene,self.particle)
        engine = ApproximateEngine(scene = self.scene)
        simulator = PhotonBatchSimulator(engine, parallel_config=ParallelConfig(n_workers=N_CORE))
        #! ERRORE FISICO? Source_points è creata sulla traccia della particella
        #Vedi Hint di ParticleInteraction
        
        scintillator = self.scene.get_scintillator()
        source_points, times = PI.generate_photon_source(x0,direction,length,DX,beta,DE_DX*scintillator.density_g_cm3,scintillator.light_yield_photons_per_MeV,rng)
        directions, photon_times = PI.photon_generator(source_points,times,scintillator.rise_time_ns,scintillator.decay_time_ns,rng)
        statuses = np.full(
            shape=times.shape,
            fill_value=PhotonStatus.RUNNING_INSIDE,
            dtype=np.int8
        )
        
        starting_batch = PhotonBatch(
            interaction_points=source_points,
            directions=directions,
            times=photon_times,
            statuses=statuses,
        )
        
        # end_batch:PhotonBatch = PE.track_photon_batch(starting_batch)
        end_batch:PhotonBatch = simulator.run_parallel(starting_batch)
        
        return starting_batch, end_batch


class MultiEvents:
    def __init__(
        self,
        scene:Scene,
        particles: Optional[Sequence[Particle]] = None,
        seed: int = 72,
        n_workers: int = N_CORE,
        parallelconfig=None
    ):
        if scene is None:
            raise ValueError("MultiEvents: scene cannot be None.")

        self.scene = scene
        self.particles = particles
        self.rng = np.random.default_rng(seed)

        engine = ApproximateEngine(scene=self.scene)
        self.simulator = PhotonBatchSimulator(
            engine,
            parallel_config=ParallelConfig(n_workers=n_workers) if parallelconfig is None else parallelconfig,
        )

    @staticmethod
    def _normalize_inputs(x0, directions, lengths, betas, particles):
        x0 = np.asarray(x0, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        lengths = np.asarray(lengths, dtype=np.float64)
        betas = np.asarray(betas, dtype=np.float64)

        if x0.ndim != 2 or x0.shape[1] != 3:
            raise ValueError("x0 must have shape (N, 3).")

        n_particles = x0.shape[0]

        if directions.ndim == 1:
            directions = np.repeat(directions[None, :], n_particles, axis=0)

        if lengths.ndim == 0:
            lengths = np.full(n_particles, float(lengths), dtype=np.float64)

        if betas.ndim == 0:
            betas = np.full(n_particles, float(betas), dtype=np.float64)

        if directions.shape != (n_particles, 3):
            raise ValueError("directions must have shape (N, 3) or (3,).")

        if lengths.shape != (n_particles,):
            raise ValueError(f"lengths must have shape (N,) or scalar, instead it has {lengths.shape}.")

        if betas.shape != (n_particles,):
            raise ValueError("betas must have shape (N,) or scalar.")

        if particles is None:
            particles = [MIP() for _ in range(n_particles)]

        if len(particles) != n_particles:
            raise ValueError("particles must have length N.")

        return x0, directions, lengths, betas, particles

    def generate_starting_batch(
        self,
        x0,
        directions=np.array([0.0, 0.0, 1.0]),
        lengths=1.0,
        betas=0.998,
        particles=None,
    ):
        
        if particles is None:
            particles = self.particles

        x0, directions, lengths, betas, particles = self._normalize_inputs(
            x0=x0,
            directions=directions,
            lengths=lengths,
            betas=betas,
            particles=particles,
        )

        scintillator = self.scene.get_scintillator()

        all_points = []
        all_directions = []
        all_times = []
        all_statuses = []

        particle_ids = []
        photon_ids = []
        point_rows = []
        initial_direction_rows = []
        distance_rows = []
        velocity_rows = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Simulating photons...", total=len(particles))

            for particle_id, particle in enumerate(particles):
                interaction = ParticleInteraction(self.scene, particle)

                progress.console.print(
                    f"Particle [yellow]{particle_id}[/yellow]:  [purple] Photon source points generation[/purple]"
                )

                source_points, emission_times = interaction.generate_photon_source(
                    x0[particle_id],
                    directions[particle_id],
                    lengths[particle_id],
                    DX,
                    betas[particle_id],
                    DE_DX * scintillator.density_g_cm3,
                    scintillator.light_yield_photons_per_MeV,
                    self.rng,
                )
                progress.console.print(
                    f"Particle [yellow]{particle_id}[/yellow]:  [blue] Photon source times generation[/blue]"
                )
                photon_directions, photon_times = interaction.photon_generator(
                    source_points,
                    emission_times,
                    scintillator.rise_time_ns,
                    scintillator.decay_time_ns,
                    self.rng,
                )
                progress.console.print(
                    f"Particle [yellow]{particle_id}[/yellow]:  [green] Update [/green]"
                )

                n_photons = len(photon_times)

                all_points.append(source_points)
                all_directions.append(photon_directions)
                all_times.append(photon_times)
                all_statuses.append(
                    np.full(
                        n_photons,
                        PhotonStatus.RUNNING_INSIDE,
                        dtype=np.int8,
                    )
                )

                particle_ids.append(
                    np.full(n_photons, particle_id, dtype=np.int64)
                )

                photon_ids.append(
                    np.arange(n_photons, dtype=np.int64)
                )

                point_rows.append(
                    np.repeat(x0[particle_id][None, :], n_photons, axis=0)
                )

                initial_direction_rows.append(
                    np.repeat(directions[particle_id][None, :], n_photons, axis=0)
                )

                distance_rows.append(
                    np.full(n_photons, lengths[particle_id], dtype=np.float64)
                )

                velocity_rows.append(
                    np.full(n_photons, betas[particle_id], dtype=np.float64)
                )
                progress.update(task, advance=1)
        
        print("\n Photon Batch generator")

        starting_batch = PhotonBatch(
            interaction_points=np.concatenate(all_points, axis=0),
            directions=np.concatenate(all_directions, axis=0),
            times=np.concatenate(all_times, axis=0),
            statuses=np.concatenate(all_statuses, axis=0),
        )

        metadata = {
            "particle_id": np.concatenate(particle_ids, axis=0),
            "particle_entry_point": np.concatenate(point_rows, axis=0),
            "particle_direction": np.concatenate(initial_direction_rows, axis=0),
            "particle_distance": np.concatenate(distance_rows, axis=0),
            "particle_beta": np.concatenate(velocity_rows, axis=0),
            "photon_id": np.concatenate(photon_ids, axis=0),
        }

        return starting_batch, metadata

    def simulate(self, *args, **kwargs):
        starting_batch, metadata = self.generate_starting_batch(*args, **kwargs)
        print("\n Simulation is running...")
        end_batch = self.simulator.run(starting_batch,parallel = False, show_progress=True)
        return starting_batch, end_batch, metadata

    def simulate_to_h5(self, filename: str, *args, **kwargs):
        starting_batch, end_batch, metadata = self.simulate(*args, **kwargs)

        datasets = {
            **metadata,

            "input_interaction_points": starting_batch.interaction_points,
            "input_directions": starting_batch.directions,
            "input_statuses": starting_batch.statuses,
            "input_times": starting_batch.times,

            "output_interaction_points": end_batch.interaction_points,
            "output_directions": end_batch.directions,
            "output_statuses": end_batch.statuses,
            "output_times": end_batch.times,
        }

        with h5py.File(filename, "w") as h5file:
            for name, values in datasets.items():
                h5file.create_dataset(name, data=np.asarray(values), chunks=True)

        return filename
    
    def simulate_to_file(
        self,
        filename: str | Path,
        *args,
        format: str | None = None,
        **kwargs,
    ):
        starting_batch, end_batch, metadata = self.simulate(*args, **kwargs)
    
        datasets = build_simulation_datasets(
            starting_batch=starting_batch,
            end_batch=end_batch,
            metadata=metadata,
        )
    
        output_format = format or infer_format(filename)
    
        if output_format == "root":
            save_datasets_to_root(datasets, filename, tree_name="photons")
        else:
            save_datasets(datasets, filename, format=output_format)
    
        return filename



def build_simulation_datasets(starting_batch, end_batch, metadata: dict) -> dict:
    return {
        **metadata,

        "input_interaction_points": starting_batch.interaction_points,
        "input_directions": starting_batch.directions,
        "input_statuses": starting_batch.statuses,
        "input_times": starting_batch.times,

        "output_interaction_points": end_batch.interaction_points,
        "output_directions": end_batch.directions,
        "output_statuses": end_batch.statuses,
        "output_times": end_batch.times,
    }


def save_datasets_to_h5(datasets: dict, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filename, "w") as h5file:
        for name, values in datasets.items():
            h5file.create_dataset(
                name,
                data=np.asarray(values),
                chunks=True,
            )


def save_datasets_to_root(
    datasets: dict,
    filename: str | Path,
    tree_name: str = "photons",
) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    data = {
        name: np.asarray(values)
        for name, values in datasets.items()
    }

    nrows = {
        name: values.shape[0]
        for name, values in data.items()
        if values.ndim > 0
    }

    if len(set(nrows.values())) != 1:
        raise ValueError(
            f"Datasets do not have the same number of rows: {nrows}"
        )

    branch_types = {}

    for name, values in data.items():
        if values.ndim == 0:
            raise ValueError(f"Scalar dataset is not supported in ROOT: {name}")

        if values.ndim == 1:
            branch_types[name] = values.dtype
        else:
            branch_types[name] = values.dtype, values.shape[1:]

    with uproot.recreate(filename) as rootfile:
        rootfile.mktree(tree_name, branch_types)
        rootfile[tree_name].extend(data)


def save_datasets_to_csv(datasets: dict, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    flat_data = {}

    for name, values in datasets.items():
        values = np.asarray(values)

        if values.ndim == 0:
            flat_data[name] = [values.item()]

        elif values.ndim == 1:
            flat_data[name] = values

        else:
            reshaped = values.reshape(values.shape[0], -1)
            for i in range(reshaped.shape[1]):
                flat_data[f"{name}_{i}"] = reshaped[:, i]

    df = pd.DataFrame(flat_data)
    df.to_csv(filename, index=False)


def save_datasets_to_npz(datasets: dict, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        filename,
        **{name: np.asarray(values) for name, values in datasets.items()}
    )


def save_datasets_to_parquet(datasets: dict, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    flat_data = {}

    for name, values in datasets.items():
        values = np.asarray(values)

        if values.ndim == 0:
            flat_data[name] = [values.item()]

        elif values.ndim == 1:
            flat_data[name] = values

        else:
            reshaped = values.reshape(values.shape[0], -1)
            for i in range(reshaped.shape[1]):
                flat_data[f"{name}_{i}"] = reshaped[:, i]

    df = pd.DataFrame(flat_data)
    df.to_parquet(filename, index=False)


def infer_format(filename: str | Path) -> str:
    suffix = Path(filename).suffix.lower()

    mapping = {
        ".h5": "h5",
        ".hdf5": "h5",
        ".root": "root",
        ".csv": "csv",
        ".npz": "npz",
        ".parquet": "parquet",
        ".pq": "parquet",
    }

    if suffix not in mapping:
        raise ValueError(f"Unsupported file extension: {suffix}")

    return mapping[suffix]


def save_datasets(
    datasets: dict,
    filename: str | Path,
    format: str | None = None,
    **kwargs,
) -> None:
    format = format or infer_format(filename)
    format = format.lower()

    savers = {
        "h5": save_datasets_to_h5,
        "hdf5": save_datasets_to_h5,
        "root": save_datasets_to_root,
        "csv": save_datasets_to_csv,
        "npz": save_datasets_to_npz,
        "parquet": save_datasets_to_parquet,
    }

    if format not in savers:
        raise ValueError(f"Unsupported format: {format}")

    savers[format](datasets, filename, **kwargs)