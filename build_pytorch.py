#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import datetime
import shutil

def setup_ccache():
    """确保 ccache 已安装并正确配置"""
    print("Setting up ccache...")
    
    # 检查是否已安装 ccache
    ccache_installed = subprocess.run("which ccache", shell=True, capture_output=True).returncode == 0
    
    if not ccache_installed:
        print("Installing ccache...")
        os_name = subprocess.check_output("awk -F= '/^NAME/{print $2}' /etc/os-release", shell=True).decode().strip()
        if "Ubuntu" in os_name:
            subprocess.run("apt-get update && apt-get install -y ccache", shell=True, check=True)
        else:
            subprocess.run("yum install -y ccache || dnf install -y ccache", shell=True, check=True)
    
    # 确保 ccache 目录存在与权限正确
    ccache_dir = os.environ.get("CCACHE_DIR", "/ccache")
    os.makedirs(ccache_dir, exist_ok=True)
    
    # 配置 ccache
    print(f"Using ccache directory: {ccache_dir}")
    subprocess.run("ccache -M 25G", shell=True)
    subprocess.run("ccache -o compression=true", shell=True)
    subprocess.run("ccache -o compression_level=6", shell=True)
    
    # 显示 ccache 统计信息
    subprocess.run("ccache -s", shell=True)
    
    return ccache_dir

def setup_cuda_env():
    """设置 CUDA 11.8 特定的环境变量（基于 build_cuda.sh）"""
    cuda_env = {
        # 基本构建配置
        "TZ": "UTC",
        "TORCH_NVCC_FLAGS": "-Xfatbin -compress-all --threads 2",
        "NCCL_ROOT_DIR": "/usr/local/cuda",
        "TH_BINARY_BUILD": "1",
        "USE_STATIC_CUDNN": "0",  # For CUDA 11.8
        "USE_STATIC_NCCL": "1",
        "ATEN_STATIC_CUDA": "1",
        "USE_CUDA_STATIC_LINK": "1",
        "INSTALL_TEST": "0",
        "USE_CUPTI_SO": "0",
        "USE_CUSPARSELT": "1",
        "USE_CUFILE": "0",  # Turned off for 11.8
        "BUILD_BUNDLE_PTXAS": "1", # Bundle ptxas into the wheel
        
        # CUDA architecture list for 11.8
        "TORCH_CUDA_ARCH_LIST": "3.5;3.7;5.0;5.2;6.0;6.1;7.0;7.5;8.0;8.6;9.0",
        
        # Package directories
        "WHEELHOUSE_DIR": "wheelhouse118",
        "LIBTORCH_HOUSE_DIR": "libtorch_house118",
        "PYTORCH_FINAL_PACKAGE_DIR": "/remote/wheelhouse118",
        
        # PyTorch 构建配置
        "USE_CUDA": "1",
        "USE_CUDNN": "1",
        "USE_MKLDNN": "1",
        "BUILD_TEST": "0",
        "USE_FBGEMM": "1",
        "BUILD_SPLIT_CUDA": "ON",
        "MAX_JOBS": "2",
        "SKIP_ALL_TESTS": "1",
        
        # 来自 build_common.sh 的其他变量
        "BUILD_PYTHONLESS": "",  # 为空因为我们正在构建 wheels
        "USE_SPLIT_BUILD": "true",  # 使用分离构建以加速编译
        "BUILD_DEBUG_INFO": "0",  # 默认不构建调试信息
        "DISABLE_RCCL": "0",  # 启用 NCCL/RCCL
        
        # 启用 ccache 进行编译加速
        "USE_CCACHE": "1",
        "CMAKE_C_COMPILER_LAUNCHER": "ccache",
        "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
        "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
    }
    
    # 应用环境变量
    for key, value in cuda_env.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    # 获取 CUDA 版本详情
    cuda_version = "11.8"
    cuda_version_nodot = "118"
    os.environ["CUDA_VERSION"] = cuda_version
    os.environ["DESIRED_CUDA"] = cuda_version_nodot
    
    # 设置依赖项打包（来自 build_cuda.sh 的 CUDA 11.8）
    setup_dependency_bundling()
    
    return cuda_version, cuda_version_nodot

def setup_dependency_bundling():
    """设置 DEPS_LIST 和 DEPS_SONAME 数组，用于将依赖项打包到 wheel 中"""
    print("Setting up dependency bundling for CUDA 11.8...")
    
    # 检测操作系统，以获取 libgomp 路径
    os_name = subprocess.check_output("awk -F= '/^NAME/{print $2}' /etc/os-release", shell=True).decode().strip()
    if "CentOS Linux" in os_name or "AlmaLinux" in os_name or "Red Hat Enterprise Linux" in os_name:
        libgomp_path = "/usr/lib64/libgomp.so.1"
    elif "Ubuntu" in os_name:
        libgomp_path = "/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    else:
        libgomp_path = "/usr/lib64/libgomp.so.1"  # 默认值
    
    # 基本依赖项
    deps_list = [libgomp_path]
    deps_soname = ["libgomp.so.1"]
    
    # 添加 CUDA 11.8 特定的依赖项
    # 对于 CUDA 11.8，我们需要在二进制文件中包含 libcusparseLt.so.0
    if os.environ.get("USE_CUSPARSELT", "1") == "1":
        deps_list.append("/usr/local/cuda/lib64/libcusparseLt.so.0")
        deps_soname.append("libcusparseLt.so.0")
    
    # 添加 CUDA 库以进行打包
    # 这些是 build_cuda.sh 为 CUDA 11.8 打包的库
    cuda_libs = [
        ("/usr/local/cuda/lib64/libcudnn_adv.so.9", "libcudnn_adv.so.9"),
        ("/usr/local/cuda/lib64/libcudnn_cnn.so.9", "libcudnn_cnn.so.9"),
        ("/usr/local/cuda/lib64/libcudnn_graph.so.9", "libcudnn_graph.so.9"),
        ("/usr/local/cuda/lib64/libcudnn_ops.so.9", "libcudnn_ops.so.9"),
        ("/usr/local/cuda/lib64/libcudnn_engines_runtime_compiled.so.9", "libcudnn_engines_runtime_compiled.so.9"),
        ("/usr/local/cuda/lib64/libcudnn_engines_precompiled.so.9", "libcudnn_engines_precompiled.so.9"),
        ("/usr/local/cuda/lib64/libcudnn_heuristic.so.9", "libcudnn_heuristic.so.9"),
        ("/usr/local/cuda/lib64/libcudnn.so.9", "libcudnn.so.9"),
        ("/usr/local/cuda/lib64/libcublas.so.11", "libcublas.so.11"),
        ("/usr/local/cuda/lib64/libcublasLt.so.11", "libcublasLt.so.11"),
        ("/usr/local/cuda/lib64/libcudart.so.11.0", "libcudart.so.11.0"),
        ("/usr/local/cuda/lib64/libnvToolsExt.so.1", "libnvToolsExt.so.1"),
        ("/usr/local/cuda/lib64/libnvrtc.so.11.2", "libnvrtc.so.11.2"),
        ("/usr/local/cuda/lib64/libnvrtc-builtins.so.11.8", "libnvrtc-builtins.so.11.8"),
    ]
    
    for lib_path, lib_soname in cuda_libs:
        deps_list.append(lib_path)
        deps_soname.append(lib_soname)
    
    # 设置为环境变量
    os.environ["DEPS_LIST"] = ";".join(deps_list)
    os.environ["DEPS_SONAME"] = ";".join(deps_soname)
    
    print(f"DEPS_LIST set with {len(deps_list)} libraries")
    print(f"DEPS_SONAME set with {len(deps_soname)} entries")

def get_manywheel_path(pytorch_version):
    """
    根据 PyTorch 版本确定 manywheel 目录的正确路径
    """
    # 检查这是否是 PyTorch 2.6 或更新版本
    major, minor = map(int, pytorch_version.split('.')[:2])
    
    if (major > 2) or (major == 2 and minor >= 6):
        # 对于 PyTorch 2.6+，manywheel 位于 .ci 目录中
        print(f"PyTorch {pytorch_version}: Using built-in manywheel directory")
        # 为脚本设置正确的权限
        subprocess.run("chmod -R +x /pytorch/.ci/manywheel/*.sh", shell=True, check=True)
        return "/pytorch/.ci/manywheel"
    else:
        # 对于旧版本，使用来自挂载路径的构建器
        print(f"PyTorch {pytorch_version}: Using mounted builder from /pytorch_builder")
        
        # 为脚本设置正确的权限
        subprocess.run("chmod -R +x /pytorch_builder/manywheel/*.sh", shell=True, check=True)
        print("Added executable permissions to build scripts")
        
        return "/pytorch_builder/manywheel"

def main():
    parser = argparse.ArgumentParser(description='Build PyTorch wheels with CUDA 11.8')
    parser.add_argument('--pytorch-version', required=True, help='PyTorch version to build')
    parser.add_argument('--python-version', required=True, help='Python version to build for')
    args = parser.parse_args()
    
    # 打印当前用户和时间戳
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Build started by user: {os.environ.get('USER', 'unknown')} at {current_time} UTC")
    print(f"Building for Python {args.python_version} and PyTorch {args.pytorch_version}")
    
    # 设置 ccache
    setup_ccache()
    
    cuda_version, cuda_version_nodot = setup_cuda_env()
    
    # 设置特定于 PyTorch 版本的变量
    os.environ["PYTORCH_BUILD_VERSION"] = args.pytorch_version
    os.environ["PYTORCH_BUILD_NUMBER"] = "1"
    os.environ["OVERRIDE_PACKAGE_VERSION"] = f"{args.pytorch_version}+cu{cuda_version_nodot}"
    
    # 设置 PYTORCH_ROOT - 这对 build_common.sh 至关重要
    os.environ["PYTORCH_ROOT"] = "/pytorch"
    print(f"Set PYTORCH_ROOT={os.environ['PYTORCH_ROOT']}")
    
    # 设置要构建的 Python 版本
    os.environ["DESIRED_PYTHON"] = args.python_version.strip()
    
    # 获取 manywheel 路径
    manywheel_path = get_manywheel_path(args.pytorch_version)
    
    # 创建脚本包装器，以确保数组正确传递到 build_common.sh
    with open("/tmp/build_wrapper.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -ex\n\n")
        
        # 导出所有环境变量
        for key, value in os.environ.items():
            # 跳过可能导致问题的环境变量
            if key in ['_', 'PWD', 'OLDPWD', 'LS_COLORS']:
                continue
            
            # 特别处理数组
            if key == "DEPS_LIST" or key == "DEPS_SONAME":
                values = value.split(";")
                f.write(f"{key}=(\n")
                for item in values:
                    f.write(f'    "{item}"\n')
                f.write(")\n")
                f.write(f"export {key}\n")
            else:
                f.write(f"export {key}=\"{value}\"\n")
        
        # 调用 build_common.sh
        f.write(f"\ncd /pytorch && {manywheel_path}/build_common.sh\n")
    
    # 使其可执行
    subprocess.run("chmod +x /tmp/build_wrapper.sh", shell=True, check=True)
    
    # 运行包装器脚本
    print(f"Running build wrapper script for Python {args.python_version}...")
    subprocess.run("/tmp/build_wrapper.sh", shell=True, check=True)
    
    print(f"\nBuild completed for Python {args.python_version}. Wheels are available in: {os.environ['PYTORCH_FINAL_PACKAGE_DIR']}")

if __name__ == "__main__":
    main()
