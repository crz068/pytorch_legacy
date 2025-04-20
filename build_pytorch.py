#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def setup_cuda_env():
    """Setup CUDA 11.8 specific environment variables based on build_cuda.sh"""
    cuda_env = {
        # Basic build config
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
        "TORCH_CUDA_ARCH_LIST": "3.5;3.7",
        
        # Package directories
        "WHEELHOUSE_DIR": "wheelhouse118",
        "LIBTORCH_HOUSE_DIR": "libtorch_house118",
        "PYTORCH_FINAL_PACKAGE_DIR": "/remote/wheelhouse118",
        
        # PyTorch build config
        "USE_CUDA": "1",
        "USE_CUDNN": "1",
        "USE_MKLDNN": "1",
        "BUILD_TEST": "0",
        "USE_FBGEMM": "1",
        "BUILD_SPLIT_CUDA": "ON",
        "MAX_JOBS": "2",
        "SKIP_ALL_TESTS": "1",
    }
    
    # Apply environment variables
    for key, value in cuda_env.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    # Get CUDA version details
    cuda_version = "11.8"
    cuda_version_nodot = "118"
    os.environ["CUDA_VERSION"] = cuda_version
    os.environ["DESIRED_CUDA"] = cuda_version_nodot
    
    return cuda_version, cuda_version_nodot

def get_manywheel_path(pytorch_version):
    """
    Determine the correct path to manywheel directory based on PyTorch version
    """
    # Check if this is PyTorch 2.6 or newer
    major, minor = map(int, pytorch_version.split('.')[:2])
    
    if (major > 2) or (major == 2 and minor >= 6):
        # For PyTorch 2.6+, manywheel is in .ci directory
        print(f"PyTorch {pytorch_version}: Using built-in manywheel directory")
        # Set the proper permissions for the scripts
        subprocess.run("chmod -R +x /pytorch/.ci/manywheel/*.sh", shell=True, check=True)
        return "/pytorch/.ci/manywheel"
    else:
        # For older versions, we need to clone pytorch/builder
        branch = f"release/{major}.{minor}"
        print(f"PyTorch {pytorch_version}: Cloning pytorch/builder repo branch {branch}")
        
        # Clone the builder repository with the correct branch
        clone_cmd = f"git clone --depth=1 -b {branch} https://github.com/pytorch/builder.git /pytorch_builder"
        print(f"Running: {clone_cmd}")
        subprocess.run(clone_cmd, shell=True, check=True)
        
        # Set the proper permissions for the scripts
        subprocess.run("chmod -R +x /pytorch_builder/manywheel/*.sh", shell=True, check=True)
        print("Added executable permissions to build scripts")
        
        return "/pytorch_builder/manywheel"

def main():
    parser = argparse.ArgumentParser(description='Build PyTorch wheels with CUDA 11.8')
    parser.add_argument('--pytorch-version', required=True, help='PyTorch version to build')
    parser.add_argument('--python-versions', default="3.9,3.10,3.11,3.12", help='Comma-separated list of Python versions')
    args = parser.parse_args()
    
    cuda_version, cuda_version_nodot = setup_cuda_env()
    
    # Set PyTorch version-specific variables
    os.environ["PYTORCH_BUILD_VERSION"] = args.pytorch_version
    os.environ["PYTORCH_BUILD_NUMBER"] = "1"
    os.environ["OVERRIDE_PACKAGE_VERSION"] = f"{args.pytorch_version}+cu{cuda_version_nodot}"
    
    # Get correct manywheel path for this PyTorch version
    manywheel_path = get_manywheel_path(args.pytorch_version)
    
    # Build for each Python version
    python_versions = args.python_versions.split(',')
    for py_version in python_versions:
        print(f"\n==== Building for Python {py_version} ====\n")
        os.environ["DESIRED_PYTHON"] = py_version.strip()
        
        # Use build_common.sh from the manywheel directory
        build_cmd = f"cd /pytorch && bash {manywheel_path}/build_common.sh"
        print(f"Running: {build_cmd}")
        subprocess.run(build_cmd, shell=True, check=True)
    
    print("\nBuild completed. Wheels are available in:", os.environ["PYTORCH_FINAL_PACKAGE_DIR"])

if __name__ == "__main__":
    main()
