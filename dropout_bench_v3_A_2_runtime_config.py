from __future__ import annotations

import os
from typing import Any, Mapping


RUNTIME_SECTION_NAME = "runtime"
CPU_CORES_KEY = "cpu_cores"
TUNING_PARALLEL_BACKEND_KEY = "tuning_parallel_backend"
MIN_CPU_CORES = 1
MAX_CPU_CORES = 8
DEFAULT_TUNING_PARALLEL_BACKEND = "processes"
ALLOWED_TUNING_PARALLEL_BACKENDS = ("processes", "threads")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def resolve_runtime_cpu_cores(shared_config: Mapping[str, Any]) -> int:
    runtime_config = shared_config.get(RUNTIME_SECTION_NAME, {})
    if runtime_config is None:
        runtime_config = {}
    if not isinstance(runtime_config, Mapping):
        raise TypeError("benchmark_shared_config.toml [runtime] must be a table when provided.")

    raw_value = runtime_config.get(CPU_CORES_KEY, MIN_CPU_CORES)
    try:
        cpu_cores = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise TypeError("benchmark_shared_config.toml [runtime].cpu_cores must be an integer.") from exc

    if cpu_cores < MIN_CPU_CORES or cpu_cores > MAX_CPU_CORES:
        raise ValueError(
            f"benchmark_shared_config.toml [runtime].cpu_cores must be between {MIN_CPU_CORES} and {MAX_CPU_CORES}."
        )

    available_cores = os.cpu_count()
    if available_cores is not None and cpu_cores > int(available_cores):
        raise ValueError(
            "benchmark_shared_config.toml [runtime].cpu_cores cannot exceed the available logical CPU count "
            f"for this host ({available_cores})."
        )
    return cpu_cores


def resolve_runtime_tuning_parallel_backend(shared_config: Mapping[str, Any]) -> str:
    runtime_config = shared_config.get(RUNTIME_SECTION_NAME, {})
    if runtime_config is None:
        runtime_config = {}
    if not isinstance(runtime_config, Mapping):
        raise TypeError("benchmark_shared_config.toml [runtime] must be a table when provided.")

    raw_value = runtime_config.get(TUNING_PARALLEL_BACKEND_KEY, DEFAULT_TUNING_PARALLEL_BACKEND)
    backend = str(raw_value).strip().lower()
    if backend not in ALLOWED_TUNING_PARALLEL_BACKENDS:
        allowed_values = ", ".join(ALLOWED_TUNING_PARALLEL_BACKENDS)
        raise ValueError(
            "benchmark_shared_config.toml [runtime].tuning_parallel_backend must be one of: "
            f"{allowed_values}."
        )
    return backend


def apply_runtime_cpu_parallelism(cpu_cores: int) -> int:
    cpu_cores = int(cpu_cores)
    for env_var in THREAD_ENV_VARS:
        os.environ[env_var] = str(cpu_cores)

    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None
    if threadpool_limits is not None:
        threadpool_limits(limits=cpu_cores)

    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        try:
            torch.set_num_threads(cpu_cores)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(cpu_cores)
        except Exception:
            pass

    return cpu_cores


def configure_runtime_cpu_cores(shared_config: Mapping[str, Any]) -> int:
    cpu_cores = resolve_runtime_cpu_cores(shared_config)
    return apply_runtime_cpu_parallelism(cpu_cores)