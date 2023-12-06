# CUDATracePreload

CUDATracePreload is a dynamic tracing tool for CUDA and NCCL API calls. It
leverages the LD\_PRELOAD mechanism to intercept and log API calls, making it
easier to capture the low level calls made by Python frameworks such as Pytorch or
Tensorflow.

## Features

- **Trace CUDA API Calls**: Automatically logs calls to CUDA APIs.
- **Monitor NCCL Operations**: Captures and reports on NCCL function invocations.
- **Easy Integration**: Simply set the LD\_PRELOAD environment variable to use.
- **Extensible**: Easy to add your own metrics

## Build

```Bash
make # creates tracer.so
```

### Build options

Several build option are available through make (e.g, `make TRACK_NCCL=1`).

```Make
LIBTORCH_CUDA_SO ?= /path/to/libtorch_cuda.so
CUDA_INCLUDE_PATH ?= /path/to/cuda/include
NCCL_INCLUDE_PATH ?= /path/to/nccl/include
TRACK_CUDA ?= 1 # Enables tracking of CUDA calls
TRACK_NCCL ?= 0 # Enables tracking of NCCL calls
```
## Running

We pass the path to tracer.so to LD\_PRELOAD. The results can be found in the
`log` folder. The number of log files may vary depending on the number of
processes per node.

```Bash
# Example for running on llama2
mkdir log
LD_PRELOAD=tracer.so python -m torch.distributed.run --nproc_per_node 2 dialog.py --ckpt_dir llama-2-13b-chat/ --tokenizer_path tokenizer.model -max_seq_len 2048 --max_batch_size 6
```

Sample log

```
[DEVICE 1 ] cudaMalloc called with arguments: devPtr = size = 165675008 
[DEVICE 1 ] cudaMalloc called with arguments: devPtr = size = 329252864 
[DEVICE 1 ] cudaMemcpyAsync called with arguments: dst = src = count = 26214400 kind = str = 
...
### Report ###
Total cudaMalloc: 18683 MB
Total ncclAllReduce count: 185139200 bytes
Total broadcast count: 0 bytes
Total reduce count: 0 bytes
Total allgather count: 8389120 bytes
Total reducescatter count: 0 bytes
Total ncclmemalloc count: 0 bytes
```

## Adding metrics

You can add metric function calls as lambda functions to the `CREATE_HOOKED_NCCL_FUNCTION` or
`CREATE_HOOKED_CUDA_FUNCTION` macros as shown below.

```C++
std::atomic<long> total_mb_allocated(0);

CREATE_HOOKED_CUDA_FUNCTION(
                cudaError_t,
                cudaMalloc,
                (void** devPtr, size_t size),
                (void**, size_t),
                (devPtr, size),
                [=]() { total_mb_allocated.fetch_add(size / 1000000, std::memory_order_relaxed); }
                )
```

Or you can define your own handler. Below is an example for `cudaMemcpy`.

```C++
cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
{
        cudaError_t (*lcudaMemcpy) ( void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind  ))dlsym(RTLD_NEXT, "cudaMemcpy");
        /* Do your own stuff */
        return lcudaMemcpy( dst, src, count, kind );
}
```



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.
