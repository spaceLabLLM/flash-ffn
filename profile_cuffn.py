import click
import torch

from cuffn_module import FFN

def do_bench(fn, warmup=25, rep=100, grad_to_none=None,
             quantiles=None,
             fast_flush=True,
             return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median"]
    import torch
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """

    # Estimate the runtime of the function
    fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    print(f"n_warmup: {n_warmup}, n_repeat: {n_repeat}")
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


def benchmark_torch_function(iters: int, function, *args, **kwargs) -> float:
    """
    function for benchmarking a pytorch function.

    Parameters
    ----------
    iters: int
        Number of iterations.
    function: lambda function
        function to benchmark.
    args: Any type
        Args to function.

    Returns
    -------
    float
        Runtime per iteration in ms.
    """
    import torch

    # Warm up
    for _ in range(5):
        function(*args, **kwargs)

    # Start benchmark.
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        function(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # in ms
    return (start_event.elapsed_time(end_event)) / iters



th_device = torch.device("cuda:0")
th_stream = torch.cuda.current_stream(th_device)

def make_torch_model(input_dim, expert_dim):  
    model = FFN(input_dim, expert_dim, False, th_device, torch.float16)
    model.half()
    model.to(th_device)
    model.eval()
    model = model.to(th_device)
    return model


def get_candidate_batch_sizes(batchsize=0):
    if batchsize == 0:
        candidate_batch_sizes = [1, 2, 4, 8, 16, 32]
    else:
        candidate_batch_sizes = [batchsize]
    return candidate_batch_sizes


def cal_gflops(B, M, K, N, with_bias=True):
    return B * (M * K * N * 2) * 2


def write_result(framework, B, M, K, N, t):
    fps = cal_gflops(B, M, K, N, with_bias=True)
    tflops = (fps * 1000) / t / 1024 / 1024 / 1024 / 1024
    prefix = "ffn (cublas gemm + cublas gemm)"
    print(
        f"{prefix} {framework}: batch_size: {B}, tokens : input_dim : expert_dim // tp_degree: {M} : {K} : {N}, {t} ms, {tflops} tflops/s",
    )
    with open(f"{prefix}_{framework}_benchmark.txt", "a") as f:
        f.write(
            f"{prefix} {framework}: batch_size: {B}, tokens : input_dim : expert_dim // tp_degree: {M} : {K} : {N}, {t} ms, {tflops} tflops/s\n"
        )


def benchmark_ffn(batchsize=0, tokens=2048, input_dim=1024, expert_dim=1024, ffn_model=None, prefix=""):
    model = ffn_model
    candidate_batch_sizes = get_candidate_batch_sizes(batchsize)

    with torch.inference_mode():
        for B in candidate_batch_sizes:
            print(f"batch_size: {B}")
            print([B*tokens, input_dim])
            x = torch.randn([B*tokens, input_dim], dtype=torch.float16, device=th_device)
            print("torch.cuda.memory_allocated", torch.cuda.memory_allocated(th_device))
            torch.cuda.synchronize()
            print("torch.cuda.synchronize()")
            y = model.forward(x)
            print("torch.cuda.memory_allocated", torch.cuda.memory_allocated(th_device))
            torch.cuda.synchronize()
            print("torch.cuda.synchronize()")
            print("do_bench")
            t = do_bench(lambda: model.forward(x), warmup=100, rep=500)
            write_result(prefix, B, tokens, input_dim, expert_dim, t)

@click.command()
@click.option(
    "--batch_size",
    type=int,
    default=0,
    help="The batch size to use for the benchmark. If 0, the batch size is default [1 : 128].",
)
@click.option(
    "--total_tokens",
    type=int,
    default=800,
)
@click.option(
    "--input_dim",
    type=int,
    default=1000,
)
@click.option(
    "--expert_dim",
    type=int,
    default=800,
)
@click.option(
    "--expert_cnt",
    type=int,
    default=8,
)
@click.option(
    "--tp_degree",
    type=int,
    default=8,
)
def benchmark(
    batch_size: int,
    total_tokens: int,
    input_dim: int,
    expert_cnt: int,
    expert_dim: int,
    tp_degree: int,
):
    tokens_per_expert = total_tokens // expert_cnt
    tp_expert_dim = expert_dim // tp_degree
    torch.cuda.empty_cache()
    alloc_mem_ptr = torch.cuda.caching_allocator_alloc(int(1e9), device=th_device, stream=th_stream)
    ffn_model = make_torch_model(input_dim=input_dim, expert_dim=tp_expert_dim)
    benchmark_ffn(batchsize=batch_size, tokens=tokens_per_expert, input_dim=input_dim, expert_dim=tp_expert_dim, ffn_model=ffn_model, prefix="ffn torch")
    torch.cuda.caching_allocator_delete(alloc_mem_ptr)
    
if __name__ == "__main__":
    torch.manual_seed(4896)
    benchmark()
