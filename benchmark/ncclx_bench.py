import argparse
import math
import sys
import torch
from torchcomms import new_comm


def do_send_recv(torchcomm, send_tensor, recv_tensor, send_rank, recv_rank, rank):
    """
    Enqueue send/recv in alternating order to avoid deadlock and measure GPU time using CUDA events.
    Returns:
        (lat_send_ms, lat_recv_ms, lat_total_ms)
    """
    stream = torch.cuda.current_stream()
    # Events
    start_total = torch.cuda.Event(enable_timing=True)
    end_total   = torch.cuda.Event(enable_timing=True)
    start_total.record(stream)

    if rank % 2 == 0:
        # Even ranks: send then recv
        torchcomm.send(send_tensor, send_rank, async_op=False)
    else:
        # Odd ranks: recv then send
        torchcomm.recv(recv_tensor, recv_rank, async_op=False)

    # Record end marker *after* both ops are enqueued on the same stream.
    end_total.record(stream)

    # Ensure GPU completion before reading elapsed times
    end_total.synchronize()

    lat_ms = start_total.elapsed_time(end_total)

    return lat_ms


def benchmark():
    parser = argparse.ArgumentParser(description="TorchComm P2P send/recv GPU-time benchmark")
    parser.add_argument("--dtype", default="float32",
                        choices=["float16", "float32", "float64", "bfloat16"],
                        help="Tensor dtype")
    parser.add_argument("--start-bytes", type=int, default=1 << 10, help="Starting message size in bytes")
    parser.add_argument("--end-bytes", type=int, default=1 << 26, help="Ending message size in bytes (inclusive cap)")
    parser.add_argument("--scale", type=float, default=2.0, help="Geometric scale factor between sizes (e.g., 2.0)")
    parser.add_argument("--iters", type=int, default=32, help="Measured iterations per message size")
    parser.add_argument("--warmup", type=int, default=32, help="Warmup iterations per message size")
    parser.add_argument("--device-backend", default="ncclx", help="TorchComm backend (default: ncclx)")
    parser.add_argument("--comm-name", default="bench_comm", help="TorchComm communicator name")
    args = parser.parse_args()

    # Map dtype string to torch dtype + element size
    dtype_map = {
        "float16": (torch.float16, 2),
        "bfloat16": (torch.bfloat16, 2),
        "float32": (torch.float32, 4),
        "float64": (torch.float64, 8),
    }
    torch_dtype, elem_size = dtype_map[args.dtype]

    # Initialize TorchComm
    device = torch.device("cuda")
    torchcomm = new_comm(args.device_backend, device, name=args.comm_name)

    rank = torchcomm.get_rank()
    world_size = torchcomm.get_size()
    if world_size < 2:
        if rank == 0:
            print("Error: world_size must be >= 2", file=sys.stderr)
        torchcomm.finalize()
        sys.exit(1)

    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(target_device)

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size

    # CSV header (each rank prints; you can grep/aggregate later)
    header_cols = [
        "world_size", "rank", "dtype", "async_op",
        "iters", "warmup",
        "elem_count", "bytes",
        "lat_ms","bw_GBps"
    ]
    if rank == 0:
        print(",".join(header_cols))

    # Size sweep
    size_bytes = max(1, args.start_bytes)
    end_cap = max(args.start_bytes, args.end_bytes)
    scale = max(args.scale, 1.0)

    while size_bytes <= end_cap + 1e-6:
        # Compute element count (round up to a whole element)
        elem_count = int(math.ceil(size_bytes / elem_size))

        # Allocate tensors on the chosen GPU
        send_tensor = torch.empty(elem_count, dtype=torch_dtype, device=target_device).fill_(float(rank))
        recv_tensor = torch.empty(elem_count, dtype=torch_dtype, device=target_device)

        # Warmup
        for _ in range(args.warmup):
            do_send_recv(torchcomm, send_tensor, recv_tensor, send_rank, recv_rank, rank)

        # Measured iterations
        lat = 0.0

        for _ in range(args.iters):
            lt = do_send_recv(torchcomm, send_tensor, recv_tensor, send_rank, recv_rank, rank)
            lat += lt

        lat_ms = lat / args.iters

        # Bytes per direction = size_bytes
        # For total (send+recv), count both directions for this rank
        seconds_total = lat_ms / 1e3

        # Guard against divide-by-zero in degenerate cases
        bw_gbps = (size_bytes / seconds_total) / (1 << 30) if seconds_total > 0 else float("inf")

        row = [
            str(world_size),
            str(rank),
            args.dtype,
            str(args.iters),
            str(args.warmup),
            str(elem_count),
            str(size_bytes),
            f"{lat_ms:.6f}",
            f"{bw_gbps:.6f}",
        ]
        if rank ==0 :
            print(",".join(row))

        # Next size
        if scale == 1.0:
            size_bytes += elem_size  # minimal linear bump by one element if scale=1
        else:
            size_bytes = int(max(size_bytes * scale, size_bytes + 1))

    torchcomm.finalize()


def main():
    benchmark()

if __name__ == "__main__":
    main()