# cuvs-playground

Baseline brute-force kNN on synthetic 100k x 128 vectors using NVIDIA cuVS.
Just to see if cuVS works as desired

## Quickstart
Realise that cuVS isn't packaged for Windows (it would be too easy if docs mentioned it, right? https://anaconda.org/rapidsai/cuvs)
Move to WSL then:
+ install micromamba (conda like package manager)
+ init micromamba shell
+ create environment (where you want it)
+ activate environment
+ install cuVS, cuPy, libcuvs (will bring CUDA toolkit if needed)
+ run sample code to see if it all works

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -p "$MAMBA_ROOT_PREFIX"
micromamba create -y -n cuvs13 python=3.13
mamba activate cuvs13
micromamba install -y -c rapidsai-nightly -c conda-forge cuvs libcuvs cupy pynvml rmm pandas cuda-version=13.0
python -m src.baseline_bruteforce