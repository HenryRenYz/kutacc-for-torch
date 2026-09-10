import os
from datetime import timedelta

import torch
import torch.distributed as dist

import kutacc_for_torch


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"rank {dist.get_rank()}: {message}")


def main() -> None:
    requested_threads = int(os.environ.get("KUTACC_TEST_TORCH_THREADS", "0"))
    if requested_threads:
        torch.set_num_threads(requested_threads)
    dist.init_process_group("kutacc", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    size = dist.get_world_size()
    if requested_threads:
        check(
            os.environ.get("KUTACC_SHM_COPY_THREADS") == str(requested_threads),
            "torch.set_num_threads did not configure the Kutacc copy workers",
        )

    dtype_names = (
        "bool", "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64",
        "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
        "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz",
    )
    dtypes = [getattr(torch, name) for name in dtype_names if hasattr(torch, name)]
    for dtype in dtypes:
        tensor = torch.empty(17, dtype=dtype)
        if rank == size - 1:
            tensor.view(torch.uint8).fill_(rank + 3)
        work = dist.broadcast(tensor, src=size - 1, async_op=True)
        work.is_completed()
        work.wait()
        check(bool(torch.all(tensor.view(torch.uint8) == rank * 0 + size + 2)), f"Broadcast failed for {dtype}")

    root = size - 1
    if rank == root:
        strided_broadcast = torch.arange(35, dtype=torch.int32).reshape(7, 5).t()
    else:
        strided_broadcast = torch.empty_strided((5, 7), (10, 1), dtype=torch.int32)
    dist.broadcast(strided_broadcast, src=root)
    expected_strided = torch.arange(35, dtype=torch.int32).reshape(7, 5).t()
    check(torch.equal(strided_broadcast, expected_strided), "mixed-stride Broadcast failed")

    mixed_dtype = torch.arange(8, dtype=torch.int32) if rank == root else torch.empty(8, dtype=torch.float32)
    dist.broadcast(mixed_dtype, src=root)
    expected_bytes = torch.arange(8, dtype=torch.int32).view(torch.uint8)
    check(torch.equal(mixed_dtype.view(torch.uint8), expected_bytes), "element-size-only Broadcast failed")

    queued_first = torch.full((31,), 11 if rank == root else 0, dtype=torch.int16)
    queued_second = torch.full((29,), 12 if rank == root else 0, dtype=torch.int16)
    first_work = dist.broadcast(queued_first, src=root, async_op=True)
    second_work = dist.broadcast(queued_second, src=root, async_op=True)
    second_work.wait()
    first_work.wait()
    check(bool(torch.all(queued_first == 11)), "first queued asynchronous Broadcast failed")
    check(bool(torch.all(queued_second == 12)), "second queued asynchronous Broadcast failed")

    source = torch.arange(24, dtype=torch.int32).reshape(4, 6).t()
    source.add_(rank * 100)
    gathered = [torch.empty_strided(source.size(), source.stride(), dtype=source.dtype) for _ in range(size)]
    dist.all_gather(gathered, source)
    for peer, tensor in enumerate(gathered):
        check(torch.equal(tensor, torch.arange(24, dtype=torch.int32).reshape(4, 6).t() + peer * 100),
              "non-contiguous AllGather failed")

    flat_input = torch.arange(7, dtype=torch.float64) + rank * 10
    flat_output = torch.empty(7 * size, dtype=torch.float64)
    dist.all_gather_into_tensor(flat_output, flat_input)
    expected = torch.cat([torch.arange(7, dtype=torch.float64) + peer * 10 for peer in range(size)])
    check(torch.equal(flat_output, expected), "AllGather single failed")

    list_inputs = [torch.full((peer + 1,), rank * 100 + peer, dtype=torch.int64) for peer in range(size)]
    list_outputs = [torch.empty(rank + 1, dtype=torch.int64) for _ in range(size)]
    work = dist.all_to_all(list_outputs, list_inputs, async_op=True)
    work.wait()
    for peer, tensor in enumerate(list_outputs):
        check(bool(torch.all(tensor == peer * 100 + rank)), "ragged AllToAll list failed")

    send_splits = [peer + 1 for peer in range(size)]
    receive_splits = [rank + 1 for _ in range(size)]
    single_input = torch.cat([
        torch.full((count,), rank * 100 + peer, dtype=torch.int32)
        for peer, count in enumerate(send_splits)
    ])
    single_output = torch.empty(sum(receive_splits), dtype=torch.int32)
    dist.all_to_all_single(single_output, single_input, receive_splits, send_splits)
    cursor = 0
    for peer, count in enumerate(receive_splits):
        check(bool(torch.all(single_output[cursor:cursor + count] == peer * 100 + rank)),
              "ragged AllToAll single failed")
        cursor += count

    ragged_input = torch.arange(rank + 1, dtype=torch.int16) + rank * 10
    counts = [peer + 1 for peer in range(size)]
    ragged_output = torch.empty(sum(counts), dtype=torch.int16)
    work = kutacc_for_torch.all_gatherv_into_tensor(
        ragged_output, ragged_input, counts, async_op=True)
    work.wait()
    expected = torch.cat([torch.arange(peer + 1, dtype=torch.int16) + peer * 10 for peer in range(size)])
    check(torch.equal(ragged_output, expected), "AllGatherV single failed")

    dist.barrier()
    if rank == 0:
        print(f"kutacc-for-torch distributed tests passed for {size} ranks; threads={torch.get_num_threads()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
