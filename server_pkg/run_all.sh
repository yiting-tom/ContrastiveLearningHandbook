#!/usr/bin/env bash
# Launch 4 parallel training lanes (2 per H100) inside the clss:train container.
# Each lane trains its method list to 200 epochs, snapshotting features at
# epochs 0,5,15,40,100,200 into /work/snaps. Logs in /work/logs.
set -e
cd "$HOME/clss"
mkdir -p logs snaps .mpl
HUID=$(id -u); HGID=$(id -g)
L=/lib/x86_64-linux-gnu
SNAP="$(seq -s, 0 200)"   # snapshot EVERY epoch (0..200)

launch() {  # $1=GPU  $2=methods  $3=lane
  docker rm -f "cl_$3" >/dev/null 2>&1 || true
  docker run -d --name "cl_$3" --ipc=host \
    --device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidiactl \
    --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
    -v $L/libcuda.so.590.48.01:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
    -v $L/libnvidia-ml.so.590.48.01:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
    -v $L/libnvidia-ptxjitcompiler.so.590.48.01:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro \
    -v $L/libnvidia-nvvm.so.590.48.01:/usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.4:ro \
    -e CUDA_VISIBLE_DEVICES=$1 -e HOME=/work -e MPLCONFIGDIR=/work/.mpl -e HUID=$HUID -e HGID=$HGID \
    -v "$HOME/clss":/work -w /work --entrypoint bash clss:train \
    -c "ldconfig; python3 train_server.py --methods $2 --data-dir data --out snaps --epochs 200 --snap $SNAP --size 96 --workers 7 > logs/$3.log 2>&1; chown -R \$HUID:\$HGID /work/snaps /work/logs" \
    >/dev/null
  echo "launched lane $3 (GPU$1): $2"
}

launch 0 "simclr_v1,byol" A
launch 0 "simclr_v2,simsiam" B
launch 1 "infomin,invariant_spread" C
launch 1 "instance_discrimination" D
echo "--- containers ---"
docker ps --format '{{.Names}}  {{.Status}}'
