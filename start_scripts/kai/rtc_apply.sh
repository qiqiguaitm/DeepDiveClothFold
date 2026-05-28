#!/bin/bash
# RTC (Real-Time Chunking) + Temporal Smoothing parameter helper.
# Run in a 2nd terminal AFTER start_autonomy.sh is up.
#
# Presets combine RTC guidance (enable_rtc + rtc_execute_horizon +
# rtc_max_guidance_weight) with temporal smoothing (inference_rate +
# latency_k + min_smooth_steps). RTC handles chunk-boundary continuity
# inside sample_actions; temporal smoothing polishes the per-step
# transition at publish time. The two layers are orthogonal.
#
# Usage:
#   ./rtc_apply.sh             # print current values
#   ./rtc_apply.sh off         # pure smoothing, no RTC guidance
#   ./rtc_apply.sh on          # JAX legacy default (3Hz, k=8, exec_h=16); aliases: default, jax_legacy
#   ./rtc_apply.sh v1_default  # V1 serve default (10Hz, latency_k=3, exec_h=6)
#   ./rtc_apply.sh jax_legacy  # same as on
#   ./rtc_apply.sh rtc_tight   # high replan + short guidance window
#   ./rtc_apply.sh rtc_long    # RTC over full horizon (A/B control)
#   ./rtc_apply.sh rtc_paper   # paper Table 4 conservative (exec_h=25, max_guid=0.5)
#   ./rtc_apply.sh rtc_paper_strong  # paper Table 4 numeric (max_guid=5.0, ⚠ 10x warning)
#   ./rtc_apply.sh rtc5        # legacy mid-freq preset (6Hz, k=4, exec_h=16)
#   ./rtc_apply.sh rtc3        # legacy alias for rtc_tight
#   ./rtc_apply.sh show        # same as no-arg
#
# Tunables (see policy_inference_node.py):
#   enable_rtc               bool  on/off RTC guidance (Pi0RTCConfig swap done at load)
#   rtc_execute_horizon      steps guidance window width [d, exec_h); V1 default 6 ≈ 2×latency_k=3
#   rtc_max_guidance_weight  float upper bound on guidance weight (paper default 0.5)
#   inference_rate           Hz    how often to query policy (V1 default 10Hz, JAX legacy 3Hz)
#   latency_k                steps drop first k of new chunk (stale-prefix trim; V1 default 3)
#   min_smooth_steps         steps floor on blend window for temporal smoothing
#   decay_alpha              0..1  legacy StreamActionBuffer param (unused in linear blend)

set -euo pipefail
NODE=/policy_inference

apply() {
    local enable_rtc="$1" exec_h="$2" max_guid="$3"
    local rate="$4" lat="$5" smooth="$6" alpha="$7"
    echo "→ enable_rtc=$enable_rtc  rtc_execute_horizon=$exec_h  rtc_max_guidance_weight=$max_guid"
    echo "  inference_rate=$rate   latency_k=$lat   min_smooth_steps=$smooth   decay_alpha=$alpha"
    ros2 param set "$NODE" enable_rtc "$enable_rtc"
    ros2 param set "$NODE" rtc_execute_horizon "$exec_h"
    ros2 param set "$NODE" rtc_max_guidance_weight "$max_guid"
    ros2 param set "$NODE" inference_rate "$rate"
    ros2 param set "$NODE" latency_k "$lat"
    ros2 param set "$NODE" min_smooth_steps "$smooth"
    ros2 param set "$NODE" decay_alpha "$alpha"
    echo "[done]"
}

mode="${1:-show}"
case "$mode" in
    off)
        # Pure temporal smoothing, RTC disabled. For A/B vs RTC-on.
        apply false 16 0.5  3.0 8 8 0.25
        ;;
    on|default|jax_legacy)
        # JAX-era default (unchanged for backward compat). Sized for 3 Hz replan
        # + 333ms cycle. Use for JAX serve (:8000) or start_autonomy_from_ckpt.sh.
        apply true  16 0.5  3.0  8 8 0.25
        ;;
    v1_default)
        # V1 default (2026-05-22 sweep M2-C, see docs/deployment/inference/rtc_implementation.md §7).
        # 10Hz replan, latency_k=3, exec_h=6 — derived from V1 forward 34ms + RTT 77ms.
        # Validated on task_a_new_pure_200_step49999: image_age -30% vs jax_legacy,
        # no GPU saturation (P50 stays 70ms), chunk MAE unchanged.
        # Use this for V1 serve (:8002) — start_autonomy_v1.sh now passes these via
        # launch args automatically, so this preset is for manual hot-tuning only.
        apply true  6  0.5  10.0 3 3 0.25
        ;;
    rtc5)
        # Legacy alias: original mid-frequency preset (predates V1).
        apply true  16 0.5  6.0  4 4 0.35
        ;;
    rtc_tight|rtc3)
        # Aggressive replan, short guidance window (for fast-changing scenes).
        apply true  12 0.8  10.0 3 3 0.45
        ;;
    rtc_long)
        # A/B control: guide over full action horizon (50 steps).
        # Maximum continuity, minimum responsiveness.
        apply true  50 0.5  3.0 8 8 0.25
        ;;
    rtc_paper)
        # Paper Table 4 alignment (conservative weight). exec_h=25 between default
        # (16) and rtc_long (50) — wider guidance window, slightly less responsive.
        # max_guid stays at 0.5 (current code's "paper default" interpretation).
        # See logs/rtc_config_compare_2026-04-29.md for the unit-mismatch discussion.
        apply true  25 0.5  3.0 8 8 0.25
        ;;
    rtc_paper_strong)
        # Paper Table 4 numeric (max_guid=5.0, 10x stronger than rtc_apply default).
        # WARNING: rtc_apply comment says "paper default 0.5"; Table 4 says 5.0.
        # Possible unit mismatch (unnormalized vs normalized weight). Verify on
        # robot incrementally — start with rtc_paper, only step up here if guidance
        # at 0.5 looks too weak (chunks visibly diverging at boundary).
        apply true  25 5.0  3.0 8 8 0.25
        ;;
    show)
        for p in enable_rtc rtc_execute_horizon rtc_max_guidance_weight \
                 inference_rate latency_k min_smooth_steps decay_alpha; do
            printf "%-28s " "$p:"
            ros2 param get "$NODE" "$p" 2>/dev/null | tail -1 || echo "(unavailable)"
        done
        ;;
    *)
        echo "Usage: $0 [off|on|rtc_tight|rtc_long|rtc5|rtc3|show]"
        exit 1
        ;;
esac
