#!/usr/bin/env bash
# Train the 6 previously-"official-weight" methods from scratch for per-epoch GIFs.
# ResNet methods @96px (GPU0), ViT methods @224px + grad-clip (GPU1). DINOv2 stays static.
set -e
cd "$HOME/clss"
mkdir -p logs snaps .mpl
HUID=$(id -u); HGID=$(id -g); L=/lib/x86_64-linux-gnu
SNAP="$(seq -s, 0 200)"

launch() {  # $1=GPU $2=methods $3=lane $4=size $5=gradclip $6=workers
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
    -c "ldconfig; python3 train_server.py --methods $2 --data-dir data --out snaps --epochs 200 --snap $SNAP --size $4 --workers $6 --grad-clip $5 > logs/$3.log 2>&1; chown -R \$HUID:\$HGID /work/snaps /work/logs" \
    >/dev/null
  echo "launched lane $3 (GPU$1, ${4}px, clip=$5): $2"
}

# GPU0: ResNet methods @96px
launch 0 "moco_v1,barlow_twins" RA 96 0 7
launch 0 "moco_v2,swav"         RB 96 0 7
# GPU1: ViT methods @224px with gradient clipping
launch 1 "moco_v3"              V1 224 1.0 8
launch 1 "dino"                 V2 224 1.0 8
echo "--- containers ---"; docker ps --filter name=cl_ --format '{{.Names}}  {{.Status}}'
