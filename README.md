# Build and Run
```bash
git clone --recursive
python3 setup.py install
python3 torch_profile_cuffn.py
 ./FlameGraph/flamegraph.pl --title "CUDA time" --countname "us." ./profiler_stacks_cuda.txt > ./perf_viz_cuda.svg
```