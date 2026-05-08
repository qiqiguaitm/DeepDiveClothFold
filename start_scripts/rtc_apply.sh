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
#   ./rtc_apply.sh on          # default: RTC + moderate smoothing
#   ./rtc_apply.sh rtc_tight   # high replan + short guidance window
#   ./rtc_apply.sh rtc_long    # RTC over full horizon (A/B control)
#   ./rtc_apply.sh rtc_paper   # paper Table 4 conservative (exec_h=25, max_guid=0.5)
#   ./rtc_apply.sh rtc_paper_strong  # paper Table 4 numeric (max_guid=5.0, ⚠ 10x warning)
#   ./rtc_apply.sh rtc5        # legacy alias for on
#   ./rtc_apply.sh rtc3        # legacy alias for rtc_tight
#   ./rtc_apply.sh show        # same as no-arg
#
# Tunables (see policy_inference_node.py):
#   enable_rtc               bool  on/off RTC guidance (Pi0RTCConfig swap done at load)
#   rtc_execute_horizon      steps guidance window width [d, exec_h); 16 ≈ 2×latency_k
#   rtc_max_guidance_weight  float upper bound on guidance weight (paper default 0.5)
#   inference_rate           Hz    how often to query policy
#   latency_k                steps drop first k of new chunk (stale-prefix trim)
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
    on|default)
        # Default: RTC with 16-step guidance window + moderate smoothing.
        apply true  16 0.5  3.0 8 8 0.25
        ;;
    rtc_tight|rtc3)
        # Aggressive replan, short guidance window (for fast-changing scenes).
        apply true  12 0.8  10.0 3 3 0.45
        ;;
    rtc5)
        # Legacy alias: same as `on` but with replan every ~5 steps.
        apply true  16 0.5  6.0 4 4 0.35
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
