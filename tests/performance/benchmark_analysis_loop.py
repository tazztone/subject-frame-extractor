import time
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

def mock_task(n):
    # Simulate some work
    time.sleep(0.01)
    return 1

def run_polling_loop(total_tasks, num_workers, batch_size):
    processed_count = 0
    futures = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while processed_count < total_tasks or futures:
            # 1. Submit new batches
            while len(futures) < num_workers and processed_count < total_tasks:
                end_idx = min(processed_count + batch_size, total_tasks)
                batch = range(processed_count, end_idx)
                futures.append(executor.submit(lambda b=batch: [mock_task(i) for i in b]))
                processed_count = end_idx

            # 2. Collect finished futures (polling)
            done = [f for f in futures if f.done()]
            futures = [f for f in futures if not f.done()]

            for future in done:
                future.result()

            if processed_count < total_tasks or futures:
                time.sleep(0.1)

    return time.time() - start_time

def run_optimized_loop(total_tasks, num_workers, batch_size):
    processed_count = 0
    futures = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while processed_count < total_tasks or futures:
            # 1. Submit new batches
            while len(futures) < num_workers and processed_count < total_tasks:
                end_idx = min(processed_count + batch_size, total_tasks)
                batch = range(processed_count, end_idx)
                futures.append(executor.submit(lambda b=batch: [mock_task(i) for i in b]))
                processed_count = end_idx

            # 2. Collect finished futures (wait)
            if futures:
                done_set, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                done = list(done_set)
                # Filter futures
                futures = [f for f in futures if f not in done_set]

                for future in done:
                    future.result()
            else:
                if processed_count < total_tasks:
                    # This case shouldn't really happen with the while condition
                    time.sleep(0.1)

    return time.time() - start_time

def main():
    total_tasks = 200
    num_workers = 4
    batch_size = 1 # Small batch size to highlight polling overhead

    print(f"Benchmarking with {total_tasks} tasks, {num_workers} workers, batch_size={batch_size}")

    polling_time = run_polling_loop(total_tasks, num_workers, batch_size)
    print(f"Polling loop time: {polling_time:.4f}s")

    optimized_time = run_optimized_loop(total_tasks, num_workers, batch_size)
    print(f"Optimized loop time: {optimized_time:.4f}s")

    improvement = (polling_time - optimized_time) / polling_time * 100
    print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    main()
