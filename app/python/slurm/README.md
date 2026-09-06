# Running the compute worker under Slurm (the HPC backend)

The application's **HPC** backend does not submit jobs itself: it connects to
a running `sirius_worker` and sends it steps to execute. On a cluster the
worker is started once per session as a Slurm job, and the application
reaches it through an SSH tunnel.

## 1. Start the worker on a node

```
SIRIUS_TOKEN=$(openssl rand -hex 16)      # keep this: the app needs it
export SIRIUS_TOKEN
sbatch app/python/slurm/sirius_worker.sbatch
```

The template asks for one GPU and eight cores; edit the `#SBATCH` lines and
the `module load` block for your cluster. It refuses to start without a
token, since the port is open to every user of the node. The log
(`sirius-worker-<jobid>.log`) prints the node name, the port (7645 by
default, `SIRIUS_PORT` to change) and the exact tunnel command.

The worker needs a Python with `numpy`; `torch` for segmentation models and
the `sirius` wheel (`pip install .` from this repository) for SIM
reconstruction on the node. Everything else in the pipeline runs where it is
implemented -- see the list the worker prints in its `hello` reply.

## 2. Tunnel the port

From your workstation:

```
ssh -N -L 7645:<node>:7645 <login-node>
```

`<node>` is the compute node from the log; the login node forwards the
connection. Leave the tunnel running for the session.

## 3. Point the application at it

Preferences ▸ HPC: host `localhost`, port `7645`, token as above. Then
choose **HPC** as the backend (Process ▸ Backend or the Backend tiles in the
parameters dock) and run a step: the application uploads the step's input
volumes, streams the worker's progress into the status bar and downloads the
result. Steps the worker cannot run (it advertises `run:<kind>` per
supported kind) report that in the log instead of failing silently.

Files referenced by parameters -- Torch models, OTFs, flat fields -- must be
readable **on the node**: give cluster paths in the parameters when the HPC
backend is selected.

## Without Slurm

The same worker runs anywhere:

```
python -m sirius_worker --host 0.0.0.0 --port 7645 --token X --device cuda
```

Locally the application starts one itself for Torch models (see
`app/python/README.md`).
