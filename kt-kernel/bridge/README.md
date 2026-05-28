## Build Example

```
cd /path/to/kt-kernel/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKTRANSFORMERS_CPU_USE_AMX_AVX512=ON
cmake --build . --target kt_bridge -j$(nproc)
```
