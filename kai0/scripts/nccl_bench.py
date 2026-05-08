"""Multi-host JAX all-reduce benchmark for gf2/gf3 cluster.

Env vars (set by launcher):
    JAX_COORDINATOR_ADDRESS  e.g. 192.168.1.2:15830
    JAX_NUM_PROCESSES        2
    JAX_PROCESS_INDEX        0 or 1
"""
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map


def main():
    coord = os.environ["JAX_COORDINATOR_ADDRESS"]
    nproc = int(os.environ["JAX_NUM_PROCESSES"])
    pid = int(os.environ["JAX_PROCESS_INDEX"])

    print(f"[init] pid={pid}/{nproc} coord={coord}", flush=True)
    t0 = time.time()
    jax.distributed.initialize(
        coordinator_address=coord, num_processes=nproc, process_id=pid
    )
    print(
        f"[ready] pid={pid} init={time.time()-t0:.1f}s "
        f"global={jax.device_count()} local={jax.local_device_count()}",
        flush=True,
    )

    n = jax.device_count()
    mesh = Mesh(np.array(jax.devices()), ("data",))

    @jax.jit
    def all_reduce(x):
        return shard_map(
            lambda y: jax.lax.psum(y, "data"),
            mesh=mesh,
            in_specs=P("data"),
            out_specs=P(),
        )(x)

    # Warm-up + benchmark several sizes. Keep small to avoid OOM during debug.
    sizes_mb = [4, 16, 64]
    if pid == 0:
        print(f"\n{'size_per_dev':>12} {'msg_total':>12} {'time_ms':>10} "
              f"{'alg_bw_GB/s':>12} {'bus_bw_GB/s':>12}", flush=True)

    for sz_mb in sizes_mb:
        elems = sz_mb * 1024 * 1024 // 4  # float32
        shape = (n, elems)
        x = jnp.ones(shape, dtype=jnp.float32)
        x = jax.device_put(x, NamedSharding(mesh, P("data")))

        # warmup
        for _ in range(3):
            all_reduce(x).block_until_ready()

        N = 20
        t = time.time()
        for _ in range(N):
            y = all_reduce(x).block_until_ready()
        dt = (time.time() - t) / N

        per_dev_bytes = elems * 4
        total_bytes = n * per_dev_bytes
        # NCCL convention: alg_bw = msg_size / time, bus_bw = alg_bw * 2(n-1)/n
        alg_bw = per_dev_bytes / dt / 1e9
        bus_bw = alg_bw * 2 * (n - 1) / n
        if pid == 0:
            print(
                f"{sz_mb:>10}MB {total_bytes/1e6:>10.0f}MB "
                f"{dt*1000:>10.2f} {alg_bw:>12.2f} {bus_bw:>12.2f}",
                flush=True,
            )

    print(f"[done] pid={pid}", flush=True)


if __name__ == "__main__":
    main()
