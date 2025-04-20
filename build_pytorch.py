#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import datetime
import glob
import shutil

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
        
        # Additional variables from build_common.sh
        "BUILD_PYTHONLESS": "",  # Empty since we're building wheels
        "USE_SPLIT_BUILD": "true",  # Use split build for faster compilation
        "BUILD_DEBUG_INFO": "0",  # Don't build debug info by default
        "DISABLE_RCCL": "0",  # Enable NCCL/RCCL
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
    
    # Setup dependency bundling (from build_cuda.sh for CUDA 11.8)
    setup_dependency_bundling()
    
    return cuda_version, cuda_version_nodot

def setup_dependency_bundling():
    """Setup the DEPS_LIST and DEPS_SONAME arrays for bundling dependencies into the wheel"""
    print("Setting up dependency bundling for CUDA 11.8...")
    
    # Detect OS for libgomp path
    os_name = subprocess.check_output("awk -F= '/^NAME/{print $2}' /etc/os-release", shell=True).decode().strip()
    if "CentOS Linux" in os_name or "AlmaLinux" in os_name or "Red Hat Enterprise Linux" in os_name:
        libgomp_path = "/usr/lib64/libgomp.so.1"
    elif "Ubuntu" in os_name:
        libgomp_path = "/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    else:
        libgomp_path = "/usr/lib64/libgomp.so.1"  # Default
    
    # Base dependencies
    deps_list = [libgomp_path]
    deps_soname = ["libgomp.so.1"]
    
    # Add CUDA 11.8 specific dependencies
    # For CUDA 11.8, we need to ship libcusparseLt.so.0 with the binary
    if os.environ.get("USE_CUSPARSELT", "1") == "1":
        deps_list.append("/usr/local/cuda/lib64/libcusparseLt.so.0")
        deps_soname.append("libcusparseLt.so.0")
    
    # Add CUDA libraries for bundling
    # These are the libraries that build_cuda.sh bundles for CUDA 11.8
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
    
    # Set as environment variables
    os.environ["DEPS_LIST"] = ";".join(deps_list)
    os.environ["DEPS_SONAME"] = ";".join(deps_soname)
    
    print(f"DEPS_LIST set with {len(deps_list)} libraries")
    print(f"DEPS_SONAME set with {len(deps_soname)} entries")

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

def ensure_wheels_are_saved():
    """Ensure wheel files are properly copied to the output directory"""
    final_package_dir = os.environ.get("PYTORCH_FINAL_PACKAGE_DIR", "/remote/wheelhouse118")
    wheelhouse_dir = os.environ.get("WHEELHOUSE_DIR", "wheelhouse118")
    
    print(f"\n=== Checking for wheel files ===")
    
    # Check internal wheel directory
    internal_wheels = glob.glob(f"/{wheelhouse_dir}/*.whl")
    if internal_wheels:
        print(f"Found {len(internal_wheels)} wheel files in internal directory:")
        for wheel in internal_wheels:
            print(f"  - {wheel}")
        
        # Copy to final package directory
        os.makedirs(final_package_dir, exist_ok=True)
        for wheel in internal_wheels:
            dest = os.path.join(final_package_dir, os.path.basename(wheel))
            print(f"Copying {wheel} to {dest}")
            shutil.copy(wheel, dest)
    
    # Check for wheels directly in /pytorch directory
    pytorch_wheels = glob.glob("/pytorch/dist/*.whl")
    if pytorch_wheels:
        print(f"Found {len(pytorch_wheels)} wheel files in PyTorch dist directory:")
        for wheel in pytorch_wheels:
            print(f"  - {wheel}")
        
        # Copy to final package directory
        os.makedirs(final_package_dir, exist_ok=True)
        for wheel in pytorch_wheels:
            dest = os.path.join(final_package_dir, os.path.basename(wheel))
            print(f"Copying {wheel} to {dest}")
            shutil.copy(wheel, dest)
    
    # Final check
    final_wheels = glob.glob(f"{final_package_dir}/*.whl")
    if final_wheels:
        print(f"Successfully saved {len(final_wheels)} wheel files to {final_package_dir}")
        for wheel in final_wheels:
            print(f"  - {wheel}")
    else:
        print(f"WARNING: No wheel files found in {final_package_dir}!")
        
        # Last resort: find all wheel files in the system
        print("Searching for wheel files across the system...")
        all_wheels = subprocess.check_output("find / -name '*.whl' 2>/dev/null || true", shell=True).decode().strip().split("\n")
        all_wheels = [w for w in all_wheels if w and "torch" in w]
        if all_wheels:
            print(f"Found {len(all_wheels)} torch wheel files in the system:")
            for wheel in all_wheels:
                print(f"  - {wheel}")
                
            # Copy to final package directory
            for wheel in all_wheels:
                if os.path.exists(wheel):
                    dest = os.path.join(final_package_dir, os.path.basename(wheel))
                    print(f"Copying {wheel} to {dest}")
                    shutil.copy(wheel, dest)

def main():
    parser = argparse.ArgumentParser(description='Build PyTorch wheels with CUDA 11.8')
    parser.add_argument('--pytorch-version', required=True, help='PyTorch version to build')
    parser.add_argument('--python-versions', default="3.9,3.10,3.11,3.12", help='Comma-separated list of Python versions')
    args = parser.parse_args()
    
    # Print current user and timestamp
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Build started by user: {os.environ.get('USER', 'unknown')} at {current_time} UTC")
    
    cuda_version, cuda_version_nodot = setup_cuda_env()
    
    # Set PyTorch version-specific variables
    os.environ["PYTORCH_BUILD_VERSION"] = args.pytorch_version
    os.environ["PYTORCH_BUILD_NUMBER"] = "1"
    os.environ["OVERRIDE_PACKAGE_VERSION"] = f"{args.pytorch_version}+cu{cuda_version_nodot}"
    
    # Set PYTORCH_ROOT - this is critical for build_common.sh
    os.environ["PYTORCH_ROOT"] = "/pytorch"
    print(f"Set PYTORCH_ROOT={os.environ['PYTORCH_ROOT']}")
    
    # Get correct manywheel path for this PyTorch version
    manywheel_path = get_manywheel_path(args.pytorch_version)
    
    # Build for each Python version
    python_versions = args.python_versions.split(',')
    for py_version in python_versions:
        print(f"\n==== Building for Python {py_version} ====\n")
        os.environ["DESIRED_PYTHON"] = py_version.strip()
        
        # Create script wrapper to make sure arrays are properly passed to build_common.sh
        with open("/tmp/build_wrapper.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -ex\n\n")
            
            # Export all environment variables
            for key, value in os.environ.items():
                # Skip some environment variables that might cause issues
                if key in ['_', 'PWD', 'OLDPWD', 'LS_COLORS']:
                    continue
                
                # Handle arrays specially
                if key == "DEPS_LIST" or key == "DEPS_SONAME":
                    values = value.split(";")
                    f.write(f"{key}=(\n")
                    for item in values:
                        f.write(f'    "{item}"\n')
                    f.write(")\n")
                    f.write(f"export {key}\n")
                else:
                    f.write(f"export {key}=\"{value}\"\n")
            
            # Call build_common.sh
            f.write(f"\ncd /pytorch && {manywheel_path}/build_common.sh\n")
        
        # Make it executable
        subprocess.run("chmod +x /tmp/build_wrapper.sh", shell=True, check=True)
        
        # Run the wrapper script
        print("Running build wrapper script...")
        try:
            subprocess.run("/tmp/build_wrapper.sh", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Build process failed with error code {e.returncode}")
            print("Attempting to save any produced wheel files...")
        
        # Ensure wheels are saved even if the build fails partially
        ensure_wheels_are_saved()
    
    # Final check to ensure all wheels are saved before exiting
    ensure_wheels_are_saved()
    
    print("\nBuild completed. Wheels are available in:", os.environ["PYTORCH_FINAL_PACKAGE_DIR"])

if __name__ == "__main__":
    main()
