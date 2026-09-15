# instant-ngp — low-VRAM fork (UVM managed memory)

这是 [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) 的魔改分支：让**小显存显卡（例如 8GB 笔记本 GPU）+ 大内存**也能训练更大的 NeRF 场景，哈希表超出显存时不再 OOM，而是自动溢到系统内存。

## 魔改内容

1. **`dependencies/patches/tcnn-gpumem-managed.patch`**
   给 tiny-cuda-nn 的 `GPUMemory` 增加 `TCNN_FORCE_MANAGED_MEMORY` 编译开关。开启后所有网络参数（NeRF 哈希表、MLP 权重、梯度）改用 `cudaMallocManaged`（CUDA 统一虚拟内存 UVM）分配：显存放不下的页面由驱动自动换出到系统 RAM、按需迁回，**不再因为哈希表超显存直接 OOM**。

2. **`CMakeLists.txt`**
   新增 CMake 选项 `NGP_BUILD_WITH_MANAGED_MEMORY`（本 fork 默认 `ON`），开启时向 tiny-cuda-nn 注入 `-DTCNN_FORCE_MANAGED_MEMORY=1`。
   关闭方式：`cmake -B build -S . -DNGP_BUILD_WITH_MANAGED_MEMORY=OFF ...`

3. **`.github/workflows/build-cuda.yml`**
   参考 Spirula 策略移植的 CI：CUDA 12.8 容器 + ccache 缓存 + 显式 `TCNN_CUDA_ARCHITECTURES`（避免无 GPU runner 检测失败）+ 关闭 GUI/OptiX。每次 push 到 `master` 自动编译并上传 Linux CUDA 二进制（sm_120 / sm_89）。

## 为什么需要它

- 原版 instant-ngp 的所有参数必须常驻显存，8GB 卡在默认 `log2_hashmap_size=19` 配置下很容易爆显存。
- UVM 让大哈希表“虚拟化”：显存装得下的部分留在显存，装不下的由驱动分页到系统内存。
- 代价：GPU 缺页时需要经 PCIe 换页，训练变慢但**能跑**。这是“用速度换容量”的取舍，适合大内存 + 小显存机器。

## 本地构建

需要 NVIDIA GPU + CUDA 12.x toolkit + CMake ≥ 3.18：

```bash
git clone --recursive https://github.com/Alpha201488/instant-ngp.git
cd instant-ngp
cmake -B build -S . -G Ninja -DNGP_BUILD_WITH_GUI=OFF -DTCNN_CUDA_ARCHITECTURES=12.0
cmake --build build -j
```

## CI 编译产物

GitHub Actions 每次 push 到 `master` 自动构建，产物在仓库 Actions 页面下载（artifact 名 `instant-ngp-lowvram-linux-cuda`，保留 30 天）。
Linux 二进制动态链接 CUDA 12.8 runtime（`libcudart.so.12`），运行机器需安装对应 CUDA driver / runtime。

## 使用

```bash
./build/instant-ngp --scene data/nerf/lego --snapshot lego.msgpack
```

进一步降低显存占用：把配置里的 `log2_hashmap_size` 调小（如 16），或配合 `--low_memory` 减少临时缓冲。

## 已知限制 / 后续方向

- v1 只提供 Linux CUDA 构建（Windows 构建后续可加）。
- UVM 分页粒度由驱动管理；后续可加 `cudaMemAdvise` 精细调优（preferred location 常驻 CPU + GPU 按需取页）。
- 训练临时工作区（GPUMemoryArena）仍使用显存，不受影响。
