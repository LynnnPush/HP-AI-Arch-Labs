#!/usr/bin/env bash
#
# GPU jax_gpu profile runner -- wraps the nsys / ncu / nvidia-smi steps for the
# jax_gpu backend (driver: profile_jax_gpu.py) into one reproducible run.
#
# Produces, in gpu_profile_results/:
#   1. nsys sweep over n_cells  -> kernel-time breakdown + CUDA launch count per N
#      (gpukernsum/cudaapisum .txt) and a launches-per-step summary table. This
#      is the launch-bound small-N region / A-crossover evidence.
#   2. nsys detailed run        -> CUDA + NVTX + sampled GPU metrics (SM occupancy
#      / DRAM BW timeline) at one n_cells; open the .nsys-rep in nsys-ui.
#   3. ncu                       -> per-kernel achieved occupancy + bandwidth
#      (needs sudo; XLA command buffers disabled; launch-count limited).
#
# Usage:
#   bash lab6_io_model/run_jax_gpu_profile.sh            # full default run
#   SMOKE=1 bash lab6_io_model/run_jax_gpu_profile.sh    # fast self-test
#   NCELLS_SWEEP="64 4096" N_SIMSTEPS=8000 bash lab6_io_model/run_jax_gpu_profile.sh
#
# Env knobs (defaults in brackets):
#   NCELLS_SWEEP ["1 2 10 30 100 1000 10000 100000"]  n_cells sweep for step 1
#                (matches sweep.py's canonical n_cells range)
#   N_SIMSTEPS   [4000]   scan trip count for the sweep
#   REPEATS      [3]      timed instances per run
#   FOCUS_NCELLS [1000]   n_cells for the detailed nsys / ncu steps
#   NCU_NSIMSTEPS[200]    short run for ncu (it replays each kernel ~6x)
#   IO_X64       [1]      float64 (matches the sweep); 0 -> f32 fast path
#   IO_RECORD_EVERY [40]  strided-recording stride (matches the sweep)
#   RUN_NSYS/RUN_NCU [1]   toggle individual steps
#   SMOKE        [unset]  shrink everything for a quick pipeline self-test
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="$REPO_ROOT/lab6_io_model/profile_jax_gpu.py"
OUT="$REPO_ROOT/lab6_io_model/gpu_profile_results"
IMPORTER="/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# ---- config ----
if [ "${SMOKE:-0}" = "1" ]; then
  NCELLS_SWEEP="${NCELLS_SWEEP:-2 100}"; N_SIMSTEPS="${N_SIMSTEPS:-400}"
  REPEATS="${REPEATS:-2}"; FOCUS_NCELLS="${FOCUS_NCELLS:-100}"
  NCU_NSIMSTEPS="${NCU_NSIMSTEPS:-40}"
else
  # Only sweep representative size of each range.
  NCELLS_SWEEP="${NCELLS_SWEEP:-10 100 1000 10000}"
  N_SIMSTEPS="${N_SIMSTEPS:-4000}"; REPEATS="${REPEATS:-2}"
  FOCUS_NCELLS="${FOCUS_NCELLS:-1000}"; NCU_NSIMSTEPS="${NCU_NSIMSTEPS:-200}"
fi
RUN_NSYS="${RUN_NSYS:-1}"; RUN_NCU="${RUN_NCU:-1}"
# Match the sweep.py jax_gpu config so the profile explains the swept curve:
# float64 + strided recording (record_every=40). Exported so every step's child
# python inherits them (the ncu step also passes them explicitly through sudo).
export IO_X64="${IO_X64:-1}" IO_RECORD_EVERY="${IO_RECORD_EVERY:-40}"

# ---- preflight: the GPU must be up (it has died on kernel-reboots before) ----
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi can't reach the driver. If the box rebooted into a new"
  echo "kernel, rebuild the driver:  sudo apt install --reinstall nvidia-driver-580-open"
  echo "  then: sudo modprobe nvidia nvidia_uvm && sudo nvidia-smi -pm 1"
  exit 1
fi
if ! python3 -c "import jax,sys; sys.exit(0 if jax.devices()[0].platform=='gpu' else 1)" 2>/dev/null; then
  echo "ERROR: JAX does not see a GPU device (falling back to CPU). Check the driver."
  exit 1
fi

echo "=================================================================="
echo "jax_gpu profile  stamp=$STAMP  focus_n=$FOCUS_NCELLS"
echo "  sweep=[$NCELLS_SWEEP]  n_simsteps=$N_SIMSTEPS  repeats=$REPEATS"
echo "  dtype=$([ "$IO_X64" = 1 ] && echo float64 || echo float32)  record_every=$IO_RECORD_EVERY  (matching the sweep)"
echo "=================================================================="

# nsys auto-import is broken on this host; convert the .qdstrm by hand.
convert() {  # $1 = report basename (no ext)
  [ -f "$1.nsys-rep" ] && return 0
  [ -f "$1.qdstrm" ] && "$IMPORTER" --input-file "$1.qdstrm" >/dev/null 2>&1
}

# ================== 1. nsys sweep: launch count + kernel time ==================
SUMMARY="$OUT/jax_summary_${STAMP}.txt"
if [ "$RUN_NSYS" = "1" ]; then
  printf "%-9s %-13s %-15s %-13s %-13s\n" "n_cells" "Mcellsteps/s" "cuLaunchKernel" "kernel_execs" "kernels/step" | tee "$SUMMARY"
  for N in $NCELLS_SWEEP; do
    base="$OUT/jax_n${N}_${STAMP}"
    log="$base.runlog"
    IO_N_CELLS="$N" IO_N_SIMSTEPS="$N_SIMSTEPS" IO_REPEATS="$REPEATS" \
      nsys profile -t cuda,nvtx -o "$base" --force-overwrite true \
      python3 "$DRIVER" >"$log" 2>&1
    convert "$base"
    nsys stats --report gpukernsum "$base.nsys-rep" >"$base.gpukernsum.txt" 2>/dev/null
    nsys stats --report cudaapisum "$base.nsys-rep" >"$base.cudaapisum.txt" 2>/dev/null
    thr=$(grep -oE "throughput=[0-9.]+" "$log" | head -1 | cut -d= -f2)
    launches=$(grep -E "cuLaunchKernel" "$base.cudaapisum.txt" | awk '{print $3}' | head -1)
    # kernels/step is robust to nsys's capture window: each body kernel fires ~once
    # per step, so sum(instances)/max(instances) = kernels launched per step (~3).
    read kexecs kstep < <(awk '$3 ~ /^[0-9]+$/ {s+=$3; if($3>m)m=$3} END{printf "%d %.2f", s, (m>0? s/m : 0)}' "$base.gpukernsum.txt")
    printf "%-9s %-13s %-15s %-13s %-13s\n" "$N" "${thr:-?}" "${launches:-?}" "${kexecs:-?}" "${kstep:-?}" | tee -a "$SUMMARY"
  done
  echo "  (per-N kernel breakdown: $OUT/jax_n<N>_${STAMP}.gpukernsum.txt)"
fi

# ============ 2. nsys detailed: GPU-metrics (occupancy / BW timeline) ==========
if [ "$RUN_NSYS" = "1" ]; then
  base="$OUT/jax_focus_n${FOCUS_NCELLS}_${STAMP}"
  echo "-- nsys detailed (+gpu-metrics) at n_cells=$FOCUS_NCELLS --"
  IO_N_CELLS="$FOCUS_NCELLS" IO_N_SIMSTEPS="$N_SIMSTEPS" IO_REPEATS="$REPEATS" \
    nsys profile -t cuda,nvtx --gpu-metrics-device=all -o "$base" \
    --force-overwrite true python3 "$DRIVER" >"$base.runlog" 2>&1
  convert "$base"
  echo "   timeline: $base.nsys-rep  (open in nsys-ui for SM-occupancy / DRAM-BW lanes)"
fi

# =============== 3. ncu: per-kernel achieved occupancy + bandwidth ============
if [ "$RUN_NCU" = "1" ]; then
  base="$OUT/jax_ncu_n${FOCUS_NCELLS}_${STAMP}"
  echo "-- ncu (sudo; command buffers off; 3 body kernels) at n_cells=$FOCUS_NCELLS --"
  # ncu needs root; sudo resets env, so pass PATH/HOME/IO_*/XLA_FLAGS explicitly.
  # --kernel-name regex:fusion targets the per-step body kernels (skips the
  # one-off wrapped_iota/broadcast setup); no -o so the metric table prints to
  # stdout (add -o "$base" for a GUI .ncu-rep as well). -c 5 (not 3) because
  # record_every>0 interleaves a recording fusion, so grab a couple extra to be
  # sure all 3 body kernels are captured.
  sudo -E env PATH="$PATH" HOME="$HOME" \
       IO_N_CELLS="$FOCUS_NCELLS" IO_N_SIMSTEPS="$NCU_NSIMSTEPS" IO_REPEATS=1 \
       IO_X64="$IO_X64" IO_RECORD_EVERY="$IO_RECORD_EVERY" \
       XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
    ncu --kernel-name "regex:fusion" -c 5 --target-processes all \
       --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum.per_second \
       python3 "$DRIVER" >"$base.txt" 2>&1
  grep -iE ", Device [0-9]|sm__warps_active|sm__throughput|dram__throughput|dram__bytes" "$base.txt" \
    | sed 's/^/   /' || true
  echo "   full ncu output: $base.txt"
fi

echo "=================================================================="
echo "done. summary -> $SUMMARY"
echo "all artifacts in $OUT (stamp $STAMP)"
echo "=================================================================="
