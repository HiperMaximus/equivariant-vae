#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="$PWD/runs/local_tmp/kaggle_kernel_$$"
  trap 'rm -rf "$TMPDIR"' EXIT
fi
mkdir -p "$TMPDIR"

# The kernel build imports eqvae (editable-installed into .venv by `uv sync`) to reuse the
# single-sourced schedule helper + patch count, so it MUST run on the venv interpreter.
# Bare python3 is the system interpreter: it has neither torch nor eqvae, and using it is
# what forced the old load-a-module-by-file-path workaround. Fail closed with an
# actionable message rather than falling back and dying deep inside a build.
build_python="${PYTHON:-.venv/bin/python}"

require_build_python() {
  # Probe EXACTLY what the build imports, not something weaker. Two traps here:
  #   -x only proves a file is executable, and a stale venv (built before the project had
  #     a [build-system], so eqvae was never installed) passes that trivially.
  #   `import eqvae` only proves src/ is reachable: src/eqvae/__init__.py is a 162-byte
  #     docstring that imports nothing, so `PYTHONPATH=src /usr/bin/python3` passes it and
  #     the build then dies at `from torch import Tensor` with a raw traceback -- the exact
  #     failure this guard exists to replace with a hint.
  # eqvae.benchmarking.__init__ pulls torch, so this probe fails closed on a torch-less
  # interpreter AND on one that cannot see eqvae at all.
  if ! "$build_python" -c 'import eqvae.benchmarking' >/dev/null 2>&1; then
    echo "error: kernel build needs a venv interpreter with eqvae AND torch importable" >&2
    echo "       tried: $build_python" >&2
    echo "hint:  uv sync --locked --python 3.12 --group dev" >&2
    exit 1
  fi
}

build_kernel_py() {
  require_build_python
  "$build_python" scripts/build_kaggle_embedded_kernel.py "$@"
}

default_kernel_dir="kaggle/kernels/non_eq_vae_debug"
default_output_dir="runs/kaggle/non_eq_vae_debug"
setup_kernel_dir="kaggle/kernels/setup_smoke"
setup_output_dir="runs/kaggle/setup_smoke"
synthetic_timing_kernel_dir="kaggle/kernels/synthetic_timing"
real_data_runtime_pretest_kernel_dir="kaggle/kernels/real_data_runtime_pretest"
real_data_runtime_pretest_output_dir="runs/kaggle/real_data_runtime_pretest"
runtime_selection_kernel_dir="kaggle/kernels/runtime_selection"
runtime_selection_output_dir="runs/kaggle/runtime_selection"
selected_runtime_debug_kernel_dir="kaggle/kernels/selected_runtime_debug"
selected_runtime_debug_output_dir="runs/kaggle/selected_runtime_debug"
selected_runtime_lr_range_kernel_dir="kaggle/kernels/selected_runtime_lr_range"
selected_runtime_lr_range_output_dir="runs/kaggle/selected_runtime_lr_range"
selected_runtime_full_kernel_dir="kaggle/kernels/selected_runtime_full"
selected_runtime_full_output_dir="runs/kaggle/selected_runtime_full"
fixed25_selector_kernel_dir="kaggle/kernels/fixed25_selector"
fixed25_selector_output_dir="runs/kaggle/fixed25_selector"
fixed25_rotation_population_kernel_dir="kaggle/kernels/fixed25_rotation_population"
fixed25_rotation_population_kernel_id="maximshtefan/eqvae-fixed25-rotation-population"
corrected_rotation_geometry_kernel_dir="kaggle/kernels/corrected_rotation_geometry"
corrected_rotation_geometry_kernel_id="maximshtefan/eqvae-corrected-rotation-geometry"
corrected_rotation_geometry_claim="runs/local/corrected_rotation_geometry_launch/push_claim.json"
decoded_transform_kernel_dir="kaggle/kernels/decoded_latent_transform"
decoded_transform_kernel_id="maximshtefan/eqvae-decoded-latent-transform"
decoded_transform_claim="runs/local/decoded_latent_transform_launch/push_claim.json"
functional_geometry_preflight_kernel_dir="kaggle/kernels/functional_geometry_preflight"
functional_geometry_preflight_kernel_id="maximshtefan/eqvae-functional-geometry-preflight-04a08ab5"
functional_geometry_preflight_claim="runs/local/functional_geometry_preflight_jvp_ladder_launch/push_claim.json"
functional_geometry_preflight_resume_kernel_dir="runs/local/functional_geometry_preflight_jvp_ladder_resume/kernel"
functional_geometry_preflight_resume_kernel_id="maximshtefan/eqvae-functional-geometry-preflight-04a08ab5-resume"
functional_geometry_preflight_resume_claim="runs/local/functional_geometry_preflight_jvp_ladder_resume/push_claim.json"
jvp_epsilon_calibration_kernel_dir="kaggle/kernels/jvp_epsilon_grid_calibration"
jvp_epsilon_calibration_kernel_id="maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5"
jvp_epsilon_calibration_claim="runs/local/jvp_epsilon_grid_calibration_launch/push_claim.json"
selected_runtime_compile_probe_kernel_dir="kaggle/kernels/selected_runtime_compile_probe"
so2_architecture_probe_kernel_dir="kaggle/kernels/so2_architecture_probe"
so2_architecture_probe_output_dir="runs/kaggle/so2_architecture_probe_v3"
so2_runtime_readiness_kernel_dir="kaggle/kernels/so2_runtime_readiness"
so2_runtime_readiness_output_dir="runs/kaggle/so2_runtime_readiness_v1"
so2_prelaunch_kernel_dir="kaggle/kernels/so2_prelaunch"
so2_prelaunch_output_dir="runs/kaggle/so2_prelaunch"
so2_full_kernel_dir="kaggle/kernels/so2_selected_runtime_full"
so2_full_output_dir="runs/kaggle/so2_selected_runtime_full"
so2_full_session1_output_dir="runs/kaggle/so2_selected_runtime_full_v1_session1"
so2_full_resume_authority_dir="runs/kaggle/so2_selected_runtime_full_session6_fresh_v1"
so2_full_resume_dataset_dir="runs/kaggle/so2_session6_resume_dataset"
so2_full_resume_dataset_slug="maximshtefan/eqvae-so2-session6-step54000"
ubc_ocean_test_atlas_kernel_dir="kaggle/kernels/ubc_ocean_test_atlas"
ubc_ocean_test_generator="kaggle/generate_ubc_ocean_test.py"
latent_inference_kernel_root="${EQVAE_LATENT_KERNEL_ROOT:-runs/local/ubc_ocean_latent_kernels}"
latent_inference_template="kaggle/kernels/ubc_ocean_latent_inference/run_template.py"
latent_finalizer_template="kaggle/kernels/ubc_ocean_latent_finalizer/run_template.py"
latent_input_bundle_dir="${EQVAE_LATENT_INPUT_BUNDLE_DIR:-runs/local/ubc_ocean_latent_input_bundle}"
latent_input_authority_dir="${EQVAE_LATENT_INPUT_AUTHORITY_DIR:-runs/local/ubc_ocean_latent_authority}"
latent_input_receipt="$latent_input_authority_dir/input_dataset_receipt.json"
latent_input_dataset_slug="maximusshtefan/eqvae-ubc-ocean-latent-inputs"
latent_resume_root="${EQVAE_LATENT_RESUME_ROOT:-runs/local/ubc_ocean_latent_resume}"
latent_ready_marker="KAGGLE_UBC_OCEAN_LATENT_INFERENCE_READY = True"
cancer_topup_plan_root="${EQVAE_CANCER_TOPUP_PLAN_ROOT:-runs/local/ubc_ocean_cancer_topup}"
cancer_topup_kernel_dir="${EQVAE_CANCER_TOPUP_KERNEL_DIR:-runs/local/ubc_ocean_cancer_topup_kernel}"
cancer_topup_receipt="${EQVAE_CANCER_TOPUP_RECEIPT:-$cancer_topup_plan_root/input_dataset_receipt.json}"
cancer_topup_ready_marker="KAGGLE_UBC_OCEAN_CANCER_TOPUP_READY = True"
mil_capacity_probe_kernel_dir="${EQVAE_MIL_CAPACITY_PROBE_KERNEL_DIR:-runs/local/ubc_ocean_mil_capacity_probe}"
mil_capacity_probe_ready_marker="KAGGLE_UBC_OCEAN_MIL_CAPACITY_PROBE_READY = True"
supervised_calibration_root="${EQVAE_SUPERVISED_CALIBRATION_ROOT:-runs/local/ubc_ocean_supervised_calibration}"
supervised_calibration_ready_marker="KAGGLE_UBC_OCEAN_SUPERVISED_CALIBRATION_READY = True"
supervised_calibration_input_root="${EQVAE_SUPERVISED_CALIBRATION_INPUT_ROOT:-runs/local/ubc_ocean_supervised_calibration_inputs}"
supervised_calibration_authority_root="${EQVAE_SUPERVISED_CALIBRATION_AUTHORITY_ROOT:-runs/local/ubc_ocean_supervised_calibration_authority}"
supervised_calibration_sweep_receipt="$supervised_calibration_authority_root/sweep_v2_input_dataset_receipt.json"
supervised_calibration_sweep_dataset_slug="maximusshtefan/eqvae-ubc-ocean-supcal-sweep-v2-inputs"
supervised_calibration_confirmation_receipt="$supervised_calibration_authority_root/confirmation_input_dataset_receipt.json"
supervised_calibration_confirmation_dataset_slug="maximusshtefan/eqvae-ubc-ocean-supcal-confirmation-inputs"
supervised_calibration_horizon_receipt="$supervised_calibration_authority_root/mil_horizon_input_dataset_receipt.json"
supervised_calibration_horizon_dataset_slug="maximusshtefan/eqvae-ubc-ocean-mil-horizon-inputs"
supervised_calibration_width128_receipt="$supervised_calibration_authority_root/mil_width128_input_dataset_receipt.json"
supervised_calibration_width128_dataset_slug="maximusshtefan/eqvae-ubc-ocean-mil-width128-inputs"
supervised_calibration_class_specific_receipt="$supervised_calibration_authority_root/mil_class_attn_input_dataset_receipt.json"
supervised_calibration_class_specific_dataset_slug="maximusshtefan/eqvae-ubc-ocean-mil-class-attn-inputs"
supervised_calibration_class_scale_receipt="$supervised_calibration_authority_root/mil_class_scale_input_dataset_receipt.json"
supervised_calibration_class_scale_dataset_slug="maximusshtefan/eqvae-ubc-ocean-mil-class-scale-inputs"
local_attention_probe_kernel_dir="kaggle/kernels/wsi45630_local_attention_probe"
local_attention_probe_kernel_id="maximusshtefan/eqvae-wsi45630-local-attention-probe"
local_attention_repair_probe_kernel_id="maximusshtefan/eqvae-wsi45630-local-attention-repair-probe"
local_attention_probe_output_dir="${EQVAE_LOCAL_ATTENTION_PROBE_OUTPUT_DIR:-runs/kaggle/wsi45630_local_attention_probe_v1}"
local_attention_probe_push_receipt="runs/local/wsi45630_local_attention_probe/push_receipt.json"
local_attention_probe_input_receipt="runs/local/wsi45630_capacity/input_receipt.json"
local_attention_repair_probe_output_dir="${EQVAE_LOCAL_ATTENTION_REPAIR_PROBE_OUTPUT_DIR:-runs/kaggle/wsi45630_local_attention_repair_probe_v2}"
local_attention_repair_probe_push_receipt="runs/local/wsi45630_local_attention_repair_probe/push_receipt.json"
local_global_capacity_kernel_dir="kaggle/kernels/wsi45630_local_global_capacity/package/kernel"
local_global_capacity_kernel_id="maximusshtefan/eqvae-wsi45630-local-global-mil-capacity"
local_global_capacity_initial_claim="runs/local/wsi45630_local_global_capacity/push_claim.json"
local_global_capacity_rejected_sources_claim="runs/local/wsi45630_local_global_capacity/retry_push_claim.json"
local_global_capacity_claim="runs/local/wsi45630_local_global_capacity/shared_access_retry_push_claim.json"
flex_attention_probe_kernel_dir="kaggle/kernels/wsi45630_flex_attention_probe"
flex_attention_probe_kernel_id="maximusshtefan/eqvae-wsi45630-flex-attention-probe"
inductor_attention_probe_kernel_dir="kaggle/kernels/wsi45630_inductor_attention_probe"
inductor_attention_probe_kernel_id="maximusshtefan/eqvae-wsi45630-inductor-attention-probe"
full_compile_probe_kernel_dir="kaggle/kernels/wsi45630_full_compile_probe/package/kernel"
full_compile_probe_kernel_id="maximusshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe"
largest_class_weighted_amp_kernel_dir="runs/local/largest_class_weighted_amp_probe/kernel"
largest_class_weighted_amp_kernel_id="maximusshtefan/eqvae-largest-class-weighted-amp-probe"
mil_training_root="runs/local/ubc_ocean_mil_training_v3"
mil_training_kernel_dir="$mil_training_root/kernel"
mil_training_kernel_id="maximusshtefan/eqvae-local-global-mil-training"
mil_training_kernel_code_file="run.py"
mil_training_dataset_slug="eqvae-local-global-mil-training-inputs-v3"
mil_training_authority_root="${EQVAE_MIL_TRAINING_AUTHORITY_ROOT:-runs/local/ubc_ocean_mil_training_authority}"
mil_training_input_receipt="${EQVAE_MIL_TRAINING_INPUT_RECEIPT:-$mil_training_authority_root/input_dataset_v3_receipt.json}"
mil_training_resume_root="${EQVAE_MIL_TRAINING_RESUME_ROOT:-runs/local/ubc_ocean_mil_training_resume}"
mil_test_root="${EQVAE_MIL_TEST_ROOT:-runs/local/ubc_ocean_mil_test_evaluation}"
mil_test_kernel_dir="$mil_test_root/kernel"
mil_test_kernel_slug="eqvae-local-global-mil-test-evaluation"
mil_test_dataset_slug="eqvae-local-global-mil-test-inputs-v1"
mil_test_input_receipt="$mil_test_root/input_dataset_receipt.json"
mil_test_launch_claim="$mil_test_root/launch_claim.json"
mil_test_accepted_reference="maximshtefan/eqvae-label-blind-mil-test-evaluation/1"
mil_test_launch_receipt_sha256="b1d7f1fd90900a0839c1b191afef5b825f8fcb464c16a4234c3e4864fa7fa243"
tissue_test_root="${EQVAE_TISSUE_TEST_ROOT:-runs/local/tissue_test_evaluation}"
tissue_test_kernel_dir="$tissue_test_root/kernel"
tissue_test_kernel_slug="eqvae-label-blind-tissue-test-evaluation"
tissue_test_dataset_slug="eqvae-tissue-test-inputs-v1"
tissue_test_input_receipt="$tissue_test_root/input_dataset_receipt.json"
tissue_test_launch_claim="$tissue_test_root/launch_claim.json"
vae_test_input_root="${EQVAE_VAE_TEST_INPUT_ROOT:-runs/local/vae_test_evaluation_input}"
vae_test_authority_root="${EQVAE_VAE_TEST_AUTHORITY_ROOT:-runs/local/vae_test_evaluation_authority}"
vae_test_kernel_dir="kaggle/kernels/vae_test_reconstruction"
vae_test_kernel_slug="eqvae-frozen-vae-test-reconstruction"
vae_test_dataset_slug="eqvae-vae-test-reconstruction-inputs-v1"
vae_test_input_receipt="$vae_test_authority_root/input_dataset_receipt.json"
vae_test_launch_claim="$vae_test_authority_root/launch_claim.json"
vae_test_accepted_reference="maximshtefan/eqvae-frozen-vae-full-test-reconstruction/1"
vae_test_launch_receipt_sha256="bd7360a9ec6831b107b7b2235cfae37060553db6434f064d949e0129788c6f3f"
tissue_fastpath_probe_root="${EQVAE_TISSUE_FASTPATH_PROBE_ROOT:-runs/local/tissue_fastpath_calibration_probe}"
tissue_fastpath_probe_kernel_dir="$tissue_fastpath_probe_root/kernel"
tissue_fastpath_probe_kernel_id="maximusshtefan/eqvae-tissue-fastpath-probe"
tissue_fastpath_probe_dataset_slug="eqvae-tissue-fastpath-probe-inputs"
tissue_fastpath_probe_authority_root="${EQVAE_TISSUE_FASTPATH_PROBE_AUTHORITY_ROOT:-runs/local/tissue_fastpath_calibration_probe_authority}"
tissue_fastpath_probe_input_receipt="$tissue_fastpath_probe_authority_root/input_dataset_receipt.json"
tissue_fastpath_probe_launch_claim="$tissue_fastpath_probe_authority_root/launch_claim.json"
tissue_training_root="${EQVAE_TISSUE_TRAINING_ROOT:-runs/local/tissue_label_efficiency_training}"
tissue_training_kernel_dir="$tissue_training_root/kernel"
tissue_training_kernel_id="maximusshtefan/eqvae-tissue-label-efficiency-training"
tissue_training_dataset_slug="eqvae-tissue-label-efficiency-training-inputs"
tissue_training_authority_root="${EQVAE_TISSUE_TRAINING_AUTHORITY_ROOT:-runs/local/tissue_label_efficiency_training_authority}"
tissue_training_input_receipt="$tissue_training_authority_root/input_dataset_receipt.json"
tissue_training_launch_claim="$tissue_training_authority_root/launch_claim.json"
tissue_training_retry_root="${EQVAE_TISSUE_TRAINING_RETRY_V2_ROOT:-runs/local/tissue_label_efficiency_training_retry_v2}"
tissue_training_retry_kernel_dir="$tissue_training_retry_root/kernel"
tissue_training_retry_authority_root="${EQVAE_TISSUE_TRAINING_RETRY_V2_AUTHORITY_ROOT:-runs/local/tissue_label_efficiency_training_retry_v2_authority}"
tissue_training_retry_launch_claim="$tissue_training_retry_authority_root/launch_claim.json"
tissue_training_retry_v3_root="${EQVAE_TISSUE_TRAINING_RETRY_V3_ROOT:-runs/local/tissue_label_efficiency_training_retry_v3}"
tissue_training_retry_v3_kernel_dir="$tissue_training_retry_v3_root/kernel"
tissue_training_retry_v3_authority_root="${EQVAE_TISSUE_TRAINING_RETRY_V3_AUTHORITY_ROOT:-runs/local/tissue_label_efficiency_training_retry_v3_authority}"
tissue_training_retry_v3_launch_claim="$tissue_training_retry_v3_authority_root/launch_claim.json"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/kaggle_kernel.sh build [kernel_dir]
  ./scripts/kaggle_kernel.sh validate [kernel_dir]
  ./scripts/kaggle_kernel.sh check [kernel_dir]
  ./scripts/kaggle_kernel.sh preflight-runtime-selection
  ./scripts/kaggle_kernel.sh preflight-fixed32-selector-readiness
  ./scripts/kaggle_kernel.sh preflight-selected-runtime-runner
  ./scripts/kaggle_kernel.sh preflight-selected-runtime-debug
  ./scripts/kaggle_kernel.sh preflight-selected-runtime-lr-range
  ./scripts/kaggle_kernel.sh preflight-selected-runtime-full
  ./scripts/kaggle_kernel.sh preflight-fixed25-selector
  ./scripts/kaggle_kernel.sh preflight-so2-architecture-probe
  ./scripts/kaggle_kernel.sh preflight-so2-runtime-readiness
  ./scripts/kaggle_kernel.sh preflight-so2-prelaunch
  ./scripts/kaggle_kernel.sh preflight-so2-selected-runtime-full
  ./scripts/kaggle_kernel.sh output-local-attention-probe
  ./scripts/kaggle_kernel.sh build-latent-inference pilot|production-all|finalizer
  ./scripts/kaggle_kernel.sh build-latent-inference resume XX
  ./scripts/kaggle_kernel.sh preflight-latent-inference pilot|production-all|run-XX|finalizer
  ./scripts/kaggle_kernel.sh build-latent-resume XX artifacts-dir
  ./scripts/kaggle_kernel.sh publish-latent-resume XX
  ./scripts/kaggle_kernel.sh verify-latent-resume XX
  ./scripts/kaggle_kernel.sh publish-latent-inputs
  ./scripts/kaggle_kernel.sh verify-latent-inputs
  ./scripts/kaggle_kernel.sh build-cancer-topup
  ./scripts/kaggle_kernel.sh preflight-cancer-topup
  ./scripts/kaggle_kernel.sh build-wsi45630-completion
  ./scripts/kaggle_kernel.sh build-full-foreground-completion
  ./scripts/kaggle_kernel.sh preflight-full-foreground-completion
  ./scripts/kaggle_kernel.sh publish-full-foreground-inputs
  ./scripts/kaggle_kernel.sh verify-full-foreground-inputs
  ./scripts/kaggle_kernel.sh output-full-foreground XX
  ./scripts/kaggle_kernel.sh publish-wsi45630-completion-inputs
  ./scripts/kaggle_kernel.sh verify-wsi45630-completion-inputs
  ./scripts/kaggle_kernel.sh publish-full-wsi-capacity-inputs
  ./scripts/kaggle_kernel.sh verify-full-wsi-capacity-inputs
  ./scripts/kaggle_kernel.sh preflight-mil-capacity-probe
  ./scripts/kaggle_kernel.sh build-tissue-training [actor]
  ./scripts/kaggle_kernel.sh validate-tissue-training [actor]
  ./scripts/kaggle_kernel.sh preflight-tissue-training [actor]
  ./scripts/kaggle_kernel.sh build-tissue-training-retry-v2 actor frozen-input-bundle
  ./scripts/kaggle_kernel.sh preflight-tissue-training-retry-v2 [actor]
  ./scripts/kaggle_kernel.sh build-tissue-training-retry-v3 actor frozen-input-bundle
  ./scripts/kaggle_kernel.sh preflight-tissue-training-retry-v3 [actor]
  ./scripts/kaggle_kernel.sh publish-tissue-training-inputs
  ./scripts/kaggle_kernel.sh verify-tissue-training-inputs
  ./scripts/kaggle_kernel.sh output-tissue-training launch-receipt.json output_dir
  ./scripts/kaggle_kernel.sh build-supervised-calibration-input sweep
  ./scripts/kaggle_kernel.sh build-supervised-calibration-input confirmation selection-audit.json
  ./scripts/kaggle_kernel.sh build-supervised-calibration-input horizon
  ./scripts/kaggle_kernel.sh build-supervised-calibration-input width128
  ./scripts/kaggle_kernel.sh build-supervised-calibration-input class_specific
  ./scripts/kaggle_kernel.sh build-supervised-calibration-input class_specific_scale_fix
  ./scripts/kaggle_kernel.sh publish-supervised-calibration-input sweep
  ./scripts/kaggle_kernel.sh publish-supervised-calibration-input confirmation selection-audit.json
  ./scripts/kaggle_kernel.sh publish-supervised-calibration-input horizon
  ./scripts/kaggle_kernel.sh publish-supervised-calibration-input width128
  ./scripts/kaggle_kernel.sh publish-supervised-calibration-input class_specific
  ./scripts/kaggle_kernel.sh publish-supervised-calibration-input class_specific_scale_fix
  ./scripts/kaggle_kernel.sh verify-supervised-calibration-input sweep
  ./scripts/kaggle_kernel.sh verify-supervised-calibration-input confirmation selection-audit.json
  ./scripts/kaggle_kernel.sh verify-supervised-calibration-input horizon
  ./scripts/kaggle_kernel.sh verify-supervised-calibration-input width128
  ./scripts/kaggle_kernel.sh verify-supervised-calibration-input class_specific
  ./scripts/kaggle_kernel.sh verify-supervised-calibration-input class_specific_scale_fix
  ./scripts/kaggle_kernel.sh build-supervised-calibration sweep
  ./scripts/kaggle_kernel.sh build-supervised-calibration confirmation selection-audit.json sweep-audit.json sweep-config.json
  ./scripts/kaggle_kernel.sh build-supervised-calibration horizon
  ./scripts/kaggle_kernel.sh build-supervised-calibration width128
  ./scripts/kaggle_kernel.sh build-supervised-calibration class_specific
  ./scripts/kaggle_kernel.sh build-supervised-calibration class_specific_scale_fix
  ./scripts/kaggle_kernel.sh preflight-supervised-calibration sweep
  ./scripts/kaggle_kernel.sh preflight-supervised-calibration confirmation selection-audit.json sweep-audit.json sweep-config.json
  ./scripts/kaggle_kernel.sh preflight-supervised-calibration horizon
  ./scripts/kaggle_kernel.sh preflight-supervised-calibration width128
  ./scripts/kaggle_kernel.sh preflight-supervised-calibration class_specific
  ./scripts/kaggle_kernel.sh preflight-supervised-calibration class_specific_scale_fix
  ./scripts/kaggle_kernel.sh build-mil-training [actor]
  ./scripts/kaggle_kernel.sh validate-mil-training [actor]
  ./scripts/kaggle_kernel.sh preflight-mil-training [actor]
  ./scripts/kaggle_kernel.sh publish-mil-training-inputs
  ./scripts/kaggle_kernel.sh verify-mil-training-inputs
  ./scripts/kaggle_kernel.sh output-mil-training launch-receipt.json output_dir
  ./scripts/kaggle_kernel.sh build-mil-test [actor]
  ./scripts/kaggle_kernel.sh validate-mil-test [actor]
  ./scripts/kaggle_kernel.sh publish-mil-test-inputs
  ./scripts/kaggle_kernel.sh status-mil-test-inputs
  ./scripts/kaggle_kernel.sh verify-mil-test-inputs
  ./scripts/kaggle_kernel.sh output-mil-test launch-receipt.json output_dir
  ./scripts/kaggle_kernel.sh score-mil-test remote_output launch-receipt.json scored_output
  ./scripts/kaggle_kernel.sh build-tissue-test [actor]
  ./scripts/kaggle_kernel.sh validate-tissue-test [actor]
  ./scripts/kaggle_kernel.sh publish-tissue-test-inputs
  ./scripts/kaggle_kernel.sh status-tissue-test-inputs
  ./scripts/kaggle_kernel.sh verify-tissue-test-inputs
  ./scripts/kaggle_kernel.sh push-tissue-test
  ./scripts/kaggle_kernel.sh output-tissue-test launch-receipt.json output_dir
  ./scripts/kaggle_kernel.sh score-tissue-test remote_output launch-receipt.json scored_output
  ./scripts/kaggle_kernel.sh validate-vae-test [actor]
  ./scripts/kaggle_kernel.sh publish-vae-test-inputs
  ./scripts/kaggle_kernel.sh status-vae-test-inputs
  ./scripts/kaggle_kernel.sh verify-vae-test-inputs
  ./scripts/kaggle_kernel.sh push-vae-test
  ./scripts/kaggle_kernel.sh output-vae-test launch-receipt.json output_dir
  ./scripts/kaggle_kernel.sh score-vae-test remote_output launch-receipt.json scored_output
  ./scripts/kaggle_kernel.sh resume-score-vae-test
  ./scripts/kaggle_kernel.sh publish-cancer-topup-inputs
  ./scripts/kaggle_kernel.sh verify-cancer-topup-inputs
  ./scripts/kaggle_kernel.sh identity
  ./scripts/kaggle_kernel.sh api-check [kernel_dir]
  ./scripts/kaggle_kernel.sh push [kernel_dir] [--wait [--wait-interval N] [--wait-max N] [--wait-queued N]]
  ./scripts/kaggle_kernel.sh status-launch launch-receipt.json
  ./scripts/kaggle_kernel.sh output-launch launch-receipt.json output_dir
  ./scripts/kaggle_kernel.sh pull-launch launch-receipt.json kernel_dir
  ./scripts/kaggle_kernel.sh dataset-download owner/dataset/version output_dir
  ./scripts/kaggle_kernel.sh status [kernel_id]
  ./scripts/kaggle_kernel.sh status-setup
  ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest
  ./scripts/kaggle_kernel.sh status-runtime-selection
  ./scripts/kaggle_kernel.sh status-selected-runtime-debug
  ./scripts/kaggle_kernel.sh status-selected-runtime-lr-range
  ./scripts/kaggle_kernel.sh status-selected-runtime-full
  ./scripts/kaggle_kernel.sh status-fixed25-selector
  ./scripts/kaggle_kernel.sh status-so2-architecture-probe
  ./scripts/kaggle_kernel.sh status-so2-runtime-readiness
  ./scripts/kaggle_kernel.sh status-so2-prelaunch
  ./scripts/kaggle_kernel.sh status-so2-selected-runtime-full
  ./scripts/kaggle_kernel.sh wait [kernel_id] [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh wait-fixed25-selector [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh wait-selected-runtime-full [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh wait-so2-architecture-probe [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh wait-so2-runtime-readiness [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh output [kernel_id] [output_dir]
  ./scripts/kaggle_kernel.sh output-setup [output_dir]
  ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest [output_dir]
  ./scripts/kaggle_kernel.sh output-runtime-selection [output_dir]
  ./scripts/kaggle_kernel.sh output-selected-runtime-debug [output_dir]
  ./scripts/kaggle_kernel.sh output-selected-runtime-lr-range [output_dir]
  ./scripts/kaggle_kernel.sh output-selected-runtime-full [output_dir]
  ./scripts/kaggle_kernel.sh output-fixed25-selector [output_dir]
  ./scripts/kaggle_kernel.sh output-so2-architecture-probe [output_dir]
  ./scripts/kaggle_kernel.sh output-so2-runtime-readiness [output_dir]
  ./scripts/kaggle_kernel.sh output-so2-prelaunch [output_dir]
  ./scripts/kaggle_kernel.sh output-so2-selected-runtime-full [output_dir]
  ./scripts/kaggle_kernel.sh pull [kernel_id] [kernel_dir]

Remote writes require KAGGLE_PUSH_CONFIRMED=1.
Spec 0036 kernel pushes additionally require KAGGLE_MIL_TRAINING_CONFIRMED=1.
Input publication additionally requires KAGGLE_DATASET_WRITE_CONFIRMED=1.
Remote writes with Kaggle source attachments also require
KAGGLE_FULL_DATASET_CONFIRMED=1.
Remote reads/downloads require KAGGLE_REMOTE_CONFIRMED=1.
Remote pulls require both KAGGLE_REMOTE_CONFIRMED=1 and
KAGGLE_PULL_CONFIRMED=1.

Generic pushes stage an account-portable copy owned by the authenticated Kaggle
user. Source dataset/kernel/model locators keep their original owners. The
accepted canonical owner/slug/version is saved under runs/local/kaggle_launches.
EOF
}

require_kaggle_cli() {
  if ! command -v kaggle >/dev/null 2>&1; then
    cat >&2 <<'EOF'
missing: kaggle

Install and authenticate the Kaggle CLI only after explicit user permission.
Do not commit Kaggle credentials.
EOF
    exit 1
  fi
}

kaggle_tool_python() {
  local kaggle_bin
  local shebang
  local interpreter
  local interpreter_command
  local interpreter_name
  local env_interpreter
  kaggle_bin="$(command -v kaggle)"
  if ! IFS= read -r shebang <"$kaggle_bin"; then
    return 1
  fi
  if [[ "$shebang" != '#!'* ]]; then
    return 1
  fi
  interpreter="${shebang#\#!}"
  interpreter_command="${interpreter%% *}"
  interpreter_name="$(basename "$interpreter_command")"
  if [[ -x "$interpreter_command" && "$interpreter_name" == python* ]]; then
    printf '%s\n' "$interpreter_command"
    return 0
  fi
  if [[ "$interpreter" == /usr/bin/env\ * ]]; then
    env_interpreter="${interpreter#/usr/bin/env }"
    if [[ "$env_interpreter" == -S\ * ]]; then
      env_interpreter="${env_interpreter#-S }"
    fi
    env_interpreter="${env_interpreter%% *}"
    if [[ "$(basename "$env_interpreter")" == python* ]] \
      && command -v "$env_interpreter" >/dev/null 2>&1; then
      command -v "$env_interpreter"
      return 0
    fi
  fi
  return 1
}

kaggle_api() {
  if [[ "${KAGGLE_DISABLE_FRESH_OAUTH:-}" != "1" \
    && -f "${HOME}/.kaggle/credentials.json" ]]; then
    local kaggle_python
    if kaggle_python="$(kaggle_tool_python)"; then
      "$kaggle_python" scripts/kaggle_oauth_exec.py "$@"
      return
    fi
    cat >&2 <<'EOF'
error: Kaggle OAuth credentials are present, but the Kaggle CLI Python
interpreter could not be resolved for the fresh-token wrapper.
Set KAGGLE_DISABLE_FRESH_OAUTH=1 only when intentionally debugging raw Kaggle
CLI authentication.
EOF
    exit 1
  fi

  kaggle "$@"
}

kaggle_auth_path_message() {
  if [[ "${KAGGLE_DISABLE_FRESH_OAUTH:-}" != "1" \
    && -f "${HOME}/.kaggle/credentials.json" ]]; then
    if kaggle_tool_python >/dev/null; then
      echo "ok: fresh OAuth wrapper selected for authenticated Kaggle calls"
      return
    fi
    cat >&2 <<'EOF'
error: Kaggle OAuth credentials are present, but the Kaggle CLI Python
interpreter could not be resolved for the fresh-token wrapper.
Set KAGGLE_DISABLE_FRESH_OAUTH=1 only when intentionally debugging raw Kaggle
CLI authentication.
EOF
    exit 1
  fi

  echo "ok: raw Kaggle auth path selected for authenticated Kaggle calls"
}

kaggle_authenticated_username() {
  require_kaggle_cli
  if [[ "${KAGGLE_DISABLE_FRESH_OAUTH:-}" == "1" \
    && -n "${KAGGLE_USERNAME:-}" ]]; then
    if [[ ! "$KAGGLE_USERNAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
      echo "error: KAGGLE_USERNAME is malformed" >&2
      exit 1
    fi
    printf '%s\n' "$KAGGLE_USERNAME"
    return
  fi
  local kaggle_python
  if ! kaggle_python="$(kaggle_tool_python)"; then
    echo "error: cannot resolve the Kaggle CLI Python interpreter" >&2
    exit 1
  fi
  if [[ "${KAGGLE_DISABLE_FRESH_OAUTH:-}" != "1" \
    && -f "${HOME}/.kaggle/credentials.json" ]]; then
    "$kaggle_python" scripts/kaggle_oauth_exec.py --print-oauth-username
    return
  fi
  "$kaggle_python" scripts/kaggle_oauth_exec.py --print-legacy-username
}

make_account_portable_kernel_snapshot() {
  local kernel_dir="$1"
  local actor="$2"
  local stage_root
  local upload_dir
  stage_root="$(mktemp -d "$TMPDIR/account_portable_kernel.XXXXXX")"
  upload_dir="$stage_root/kernel"
  require_build_python
  "$build_python" -m eqvae.kaggle_resources snapshot \
    --source-dir "$kernel_dir" \
    --destination-dir "$upload_dir" \
    --actor "$actor" >/dev/null
  printf '%s\n' "$upload_dir"
}

make_corrected_rotation_geometry_snapshot() {
  local kernel_dir="$1"
  local actor="$2"
  local stage_root
  local source_dir
  local upload_dir
  stage_root="$(mktemp -d "$TMPDIR/corrected_rotation_geometry.XXXXXX")"
  source_dir="$stage_root/source"
  upload_dir="$stage_root/kernel"
  mkdir -p "$source_dir"
  cp -- "$kernel_dir/kernel-metadata.json" "$source_dir/kernel-metadata.json"
  cp -- "$kernel_dir/run.py" "$source_dir/run.py"
  require_build_python
  "$build_python" -m eqvae.kaggle_resources snapshot \
    --source-dir "$source_dir" \
    --destination-dir "$upload_dir" \
    --actor "$actor" >/dev/null
  printf '%s\n' "$upload_dir"
}

record_account_portable_launch() {
  local kernel_dir="$1"
  local upload_dir="$2"
  local accepted_reference="$3"
  local receipt_root="${EQVAE_KAGGLE_LAUNCH_RECEIPT_ROOT:-runs/local/kaggle_launches}"
  local args=(
    -m eqvae.kaggle_resources receipt
    --source-dir "$kernel_dir"
    --upload-dir "$upload_dir"
    --receipt-root "$receipt_root"
    --accepted-reference "$accepted_reference"
  )
  require_build_python
  "$build_python" "${args[@]}"
}

claim_corrected_rotation_geometry_push() {
  local kernel_dir="$1"
  local upload_dir="$2"
  local actor="$3"
  mkdir -p "$(dirname "$corrected_rotation_geometry_claim")"
  python3 - "$corrected_rotation_geometry_claim" "$kernel_dir" "$upload_dir" \
    "$actor" <<'PYSPEC0050CLAIM'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

claim, source_dir, upload_dir = map(Path, sys.argv[1:4])
actor = sys.argv[4]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(directory):
    entries = list(directory.iterdir())
    if {path.name for path in entries} != {"kernel-metadata.json", "run.py"}:
        raise SystemExit("Spec 0050 upload snapshot allow-list differs")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise SystemExit("Spec 0050 upload snapshot contains a non-regular file")
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(entries)
    }


if actor != "maximshtefan":
    raise SystemExit("Spec 0050 actor differs")
source_files = {
    name: {
        "bytes": (source_dir / name).stat().st_size,
        "sha256": sha256(source_dir / name),
    }
    for name in ("kernel-metadata.json", "run.py")
}

upload_files = inventory(upload_dir)
payload = {
    "schema_version": "spec0050.push_attempt.v1",
    "authorization": "spec0050_corrected_rotation_geometry_authorized",
    "authority_consumed": True,
    "status": "attempt_claimed",
    "actor": actor,
    "kernel_id": "maximshtefan/eqvae-corrected-rotation-geometry",
    "source_files": source_files,
    "upload_files": upload_files,
    "attempt_started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
with claim.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory = os.open(claim.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0050CLAIM
}

claim_decoded_transform_push() {
  local kernel_dir="$1"
  local upload_dir="$2"
  local actor="$3"
  mkdir -p "$(dirname "$decoded_transform_claim")"
  python3 - "$decoded_transform_claim" "$kernel_dir" "$upload_dir" \
    "$actor" <<'PYSPEC0051CLAIM'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

claim, source_dir, upload_dir = map(Path, sys.argv[1:4])
actor = sys.argv[4]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(directory):
    entries = list(directory.iterdir())
    if {path.name for path in entries} != {"kernel-metadata.json", "run.py"}:
        raise SystemExit("Spec 0051 upload snapshot allow-list differs")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise SystemExit("Spec 0051 upload snapshot contains a non-regular file")
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(entries)
    }


if actor != "maximshtefan":
    raise SystemExit("Spec 0051 actor differs")
source_files = {
    name: {
        "bytes": (source_dir / name).stat().st_size,
        "sha256": sha256(source_dir / name),
    }
    for name in ("kernel-metadata.json", "run.py")
}
payload = {
    "schema_version": "spec0051.push_attempt.v1",
    "authorization": "spec0051_decoded_latent_transform_authorized",
    "authority_consumed": True,
    "status": "attempt_claimed",
    "actor": actor,
    "kernel_id": "maximshtefan/eqvae-decoded-latent-transform",
    "source_files": source_files,
    "upload_files": inventory(upload_dir),
    "attempt_started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
with claim.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory = os.open(claim.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0051CLAIM
}

claim_functional_geometry_preflight_push() {
  local kernel_dir="$1"
  local upload_dir="$2"
  local actor="$3"
  mkdir -p "$(dirname "$functional_geometry_preflight_claim")"
  python3 - "$functional_geometry_preflight_claim" "$kernel_dir" "$upload_dir" \
    "$actor" <<'PYSPEC0053PREFLIGHTCLAIM'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

claim, source_dir, upload_dir = map(Path, sys.argv[1:4])
actor = sys.argv[4]


def inventory(directory, expected):
    entries = [
        path for path in directory.iterdir() if path.is_file() or path.is_symlink()
    ]
    if {path.name for path in entries} != expected:
        raise SystemExit("Spec 0057 JVP ladder upload snapshot allow-list differs")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise SystemExit("Spec 0057 JVP ladder upload snapshot is not regular")
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(entries)
    }


if actor != "maximshtefan":
    raise SystemExit("Spec 0057 JVP ladder actor differs")
source_files = inventory(
    source_dir,
    {"kernel-metadata.json", "run.py", "run_template.py"},
)
payload = {
    "schema_version": "spec0057.jvp_epsilon_ladder_push_attempt.v1",
    "authorization": "spec0057_jvp_epsilon_ladder_authorized",
    "authority_consumed": True,
    "status": "attempt_claimed",
    "actor": actor,
    "kernel_id": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5",
    "source_files": source_files,
    "upload_files": inventory(upload_dir, {"kernel-metadata.json", "run.py"}),
    "attempt_started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
with claim.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory = os.open(claim.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0053PREFLIGHTCLAIM
}

claim_jvp_epsilon_calibration_push() {
  local kernel_dir="$1"
  local upload_dir="$2"
  local actor="$3"
  mkdir -p "$(dirname "$jvp_epsilon_calibration_claim")"
  python3 - "$jvp_epsilon_calibration_claim" "$kernel_dir" "$upload_dir" \
    "$actor" <<'PYSPEC0058CLAIM'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

claim, source_dir, upload_dir = map(Path, sys.argv[1:4])
actor = sys.argv[4]


def inventory(directory, expected):
    entries = list(directory.iterdir())
    if {path.name for path in entries} != expected:
        raise SystemExit("Spec 0058 calibration snapshot allow-list differs")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise SystemExit("Spec 0058 calibration snapshot is not regular")
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(entries)
    }


if actor != "maximshtefan":
    raise SystemExit("Spec 0058 calibration actor differs")
payload = {
    "schema_version": "spec0058.jvp_epsilon_grid_calibration_push_attempt.v1",
    "authorization": "spec0058_jvp_epsilon_grid_calibration_authorized",
    "authority_consumed": True,
    "status": "attempt_claimed",
    "actor": actor,
    "kernel_id": "maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5",
    "source_files": inventory(
        source_dir,
        {"kernel-metadata.json", "run.py", "run_template.py"},
    ),
    "upload_files": inventory(upload_dir, {"kernel-metadata.json", "run.py"}),
    "attempt_started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
with claim.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory = os.open(claim.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0058CLAIM
}

claim_functional_geometry_preflight_resume_push() {
  local kernel_dir="$1"
  local upload_dir="$2"
  local actor="$3"
  mkdir -p "$(dirname "$functional_geometry_preflight_resume_claim")"
  python3 - "$functional_geometry_preflight_resume_claim" "$kernel_dir" \
    "$upload_dir" "$actor" <<'PYSPEC0057RESUMECLAIM'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

claim, source_dir, upload_dir = map(Path, sys.argv[1:4])
actor = sys.argv[4]


def inventory(directory, expected):
    entries = list(directory.iterdir())
    if {path.name for path in entries} != expected:
        raise SystemExit("Spec 0057 resume upload inventory differs")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise SystemExit("Spec 0057 resume upload contains a non-regular file")
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(entries)
    }


if actor != "maximshtefan":
    raise SystemExit("Spec 0057 resume actor differs")
contract = json.loads((source_dir / "continuation_contract.json").read_text(encoding="utf-8"))
expected_binding = {
    "schema": "spec0057.preflight_continuation_binding.v1",
    "parent_kernel_id": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5",
    "parent_version": 1,
    "parent_reference": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5/1",
    "parent_contract_sha256": "bc48f1f6d4a3088501054aa246cfa2785adec9947914faf76e9b44501e359482",
    "parent_spec_sha256": "6821036a7b616f34ec5ddee339d13bc15b8167b0359e089dbc89a5eed30144b0",
    "pending_work_id": "a8a3d5842f7d9e9f3be45029f6ed55d2e8e6651e2476312c151211259010c472",
    "child_kernel_id": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5-resume",
}
for field, value in expected_binding.items():
    if contract.get(field) != value:
        raise SystemExit(f"Spec 0057 resume binding differs at {field}")
manifest_hash = contract.get("parent_manifest_sha256")
if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
    raise SystemExit("Spec 0057 resume parent manifest hash differs")
output_receipt_hash = contract.get("parent_output_receipt_sha256")
if not isinstance(output_receipt_hash, str) or len(output_receipt_hash) != 64:
    raise SystemExit("Spec 0057 resume parent output receipt hash differs")
payload = {
    "schema_version": "spec0057.jvp_epsilon_ladder_resume_push_attempt.v1",
    "authorization": "spec0057_jvp_epsilon_ladder_authorized",
    "authority_consumed": True,
    "status": "attempt_claimed",
    "actor": actor,
    "kernel_id": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5-resume",
    "parent_reference": contract["parent_reference"],
    "parent_manifest_sha256": manifest_hash,
    "source_files": inventory(source_dir, {"continuation_contract.json", "kernel-metadata.json", "run.py"}),
    "upload_files": inventory(upload_dir, {"kernel-metadata.json", "run.py"}),
    "attempt_started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
with claim.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory = os.open(claim.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0057RESUMECLAIM
}

preflight_functional_geometry_slug() {
  local kernel_id="$1"
  local kernel_slug="${kernel_id#*/}"
  local listing
  require_remote_confirmed
  listing="$(kaggle_api kernels list --mine --search "$kernel_slug" --csv)"
  KAGGLE_SPEC0053_KERNEL_LISTING="$listing" python3 - "$kernel_id" <<'PYSPEC0053UNLAUNCHED'
import csv
import io
import os
import sys

kernel_id = sys.argv[1].casefold()
rows = csv.reader(io.StringIO(os.environ["KAGGLE_SPEC0053_KERNEL_LISTING"]))
if any(any(cell.casefold() == kernel_id for cell in row) for row in rows):
    raise SystemExit("Spec 0053 unique kernel slug already exists remotely")
PYSPEC0053UNLAUNCHED
}

claim_local_global_capacity_push() {
  local kernel_dir="$1"
  local upload_dir="$2"
  local actor="$3"
  require_build_python
  "$build_python" - "$kernel_dir" "$upload_dir" "$actor" \
    "$local_global_capacity_claim" <<'PYLOCALGLOBALCAPACITYCLAIM'
import hashlib
import json
import os
import sys
from pathlib import Path

source_dir, upload_dir, actor, destination = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    sys.argv[3],
    Path(sys.argv[4]),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_metadata = json.loads(
    (source_dir / "kernel-metadata.json").read_text(encoding="utf-8"),
)
upload_metadata = json.loads(
    (upload_dir / "kernel-metadata.json").read_text(encoding="utf-8"),
)
if upload_metadata.get("id") != f"{actor}/eqvae-wsi45630-local-global-mil-capacity":
    raise SystemExit("portable Spec 0030 owner/slug differs")
for field in (
    "dataset_sources",
    "kernel_sources",
    "model_sources",
    "competition_sources",
):
    if upload_metadata.get(field) != source_metadata.get(field):
        raise SystemExit("portable Spec 0030 source locator changed")
entries = list(source_dir.iterdir())
if {path.name for path in entries} != {"kernel-metadata.json", "run.py"} or any(
    path.is_symlink() or not path.is_file() for path in entries
):
    raise SystemExit("Spec 0030 package allow-list differs at claim")
upload_entries = list(upload_dir.iterdir())
if {path.name for path in upload_entries} != {"kernel-metadata.json", "run.py"} or any(
    path.is_symlink() or not path.is_file() for path in upload_entries
):
    raise SystemExit("Spec 0030 upload snapshot allow-list differs at claim")
files = {
    path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
    for path in sorted(entries)
}
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x", encoding="utf-8") as handle:
    json.dump(
        {
            "schema_version": "spec0030.capacity_shared_access_retry_claim.v1",
            "authorization": "spec0030_local_global_capacity_shared_access_retry_authorized",
            "authority_consumed": True,
            "actor": actor,
            "requested_kernel_id": upload_metadata["id"],
            "source_locators": {
                field: upload_metadata.get(field, [])
                for field in (
                    "dataset_sources",
                    "kernel_sources",
                    "model_sources",
                    "competition_sources",
                )
            },
            "source_files": files,
            "upload_metadata_sha256": sha256(upload_dir / "kernel-metadata.json"),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYLOCALGLOBALCAPACITYCLAIM
}

confirmed_kernel_reference() {
  local response="$1"
  require_build_python
  printf '%s\n' "$response" \
    | "$build_python" -m eqvae.kaggle_resources confirmation
}

kernel_reference_from_launch_receipt() {
  local receipt="$1"
  require_build_python
  "$build_python" - "$receipt" <<'PYKAGGLELAUNCHREF'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
reference = payload.get("kernel_reference")
kernel_id = payload.get("kernel_id")
actor = payload.get("actor")
version = payload.get("accepted_version")
if (
    payload.get("schema_version") != "eqvae.kaggle_kernel_launch.v1"
    or not isinstance(reference, str)
    or not isinstance(kernel_id, str)
    or not isinstance(actor, str)
    or isinstance(version, bool)
    or not isinstance(version, int)
    or version < 1
    or reference != f"{kernel_id}/{version}"
    or kernel_id.count("/") != 1
    or kernel_id.split("/", 1)[0] != actor
):
    raise SystemExit("invalid account-portable Kaggle launch receipt")
print(reference)
PYKAGGLELAUNCHREF
}

validated_versioned_reference() {
  local reference="$1"
  require_build_python
  "$build_python" -m eqvae.kaggle_resources validate-versioned-reference \
    --reference "$reference"
}

record_kaggle_download() {
  local resource_kind="$1"
  local resource_reference="$2"
  local download_dir="$3"
  local receipt_name="$4"
  require_build_python
  "$build_python" -m eqvae.kaggle_resources download-receipt \
    --resource-kind "$resource_kind" \
    --resource-reference "$resource_reference" \
    --download-dir "$download_dir" \
    --receipt-name "$receipt_name"
}

require_remote_confirmed() {
  if [[ "${KAGGLE_REMOTE_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_REMOTE_CONFIRMED=1 after explicit user permission" >&2
    exit 1
  fi
}

wait_kernel_until_settled() {
  # Poll a kernel's status until it leaves the actively-pending states, then
  # print WAIT_SETTLED_STATUS=<outcome> and return so the caller is woken instead
  # of hanging. The two pending states are bounded separately:
  #   * RUNNING -- polled at the slow steady cadence (poll_interval) and bounded
  #     by max_polls; on exhaustion it prints TIMEOUT_STILL_PENDING (return 2).
  #   * QUEUED  -- polled faster and bounded by a shorter budget
  #     (max_queued_seconds); a kernel that never gets a compute slot is
  #     abandoned early with QUEUED_TIMEOUT (return 3) rather than tying the
  #     watcher up for the full multi-hour running backstop.
  # Every other status settles and returns 0: COMPLETE, ERROR, a cancellation
  # (CANCEL_REQUESTED / CANCEL_ACKNOWLEDGED, e.g. the Kaggle session time limit
  # killing the run), or any unrecognized status. Each poll goes through
  # kaggle_api, which mints a fresh OAuth token, so multi-hour waits stay
  # authenticated. A transient unparseable reply is retried against max_polls.
  # Every path wakes the caller.
  local kernel_id="$1"
  # Default to a 5-minute steady cadence: most watched kernels are multi-hour
  # runs, so a slow poll is plenty and stays far clear of API rate limits.
  local poll_interval="${2:-300}"
  local max_polls="${3:-180}"
  # Abandon a kernel stuck in QUEUED after this many seconds (default 5 min).
  local max_queued_seconds="${4:-300}"
  # Hard floor so no argument can hammer the Kaggle API into rate limiting.
  if ((poll_interval < 10)); then
    echo "wait: clamping poll interval to the 10s minimum (was ${poll_interval}s)" >&2
    poll_interval=10
  fi
  # Poll QUEUED faster than the steady cadence so the shorter queue budget is
  # enforced with useful granularity, but never below the floor or above the
  # steady interval.
  local queued_interval=30
  if ((queued_interval > poll_interval)); then
    queued_interval="$poll_interval"
  fi
  if ((queued_interval < 10)); then
    queued_interval=10
  fi
  local running_polls=0 queued_elapsed=0 status status_line
  while :; do
    status_line="$(kaggle_api kernels status "$kernel_id" 2>&1)" || true
    status="$(printf '%s\n' "$status_line" \
      | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1 || true)"
    status="${status#KernelWorkerStatus.}"
    case "$status" in
    QUEUED)
      echo "wait: QUEUED ${queued_elapsed}s/${max_queued_seconds}s"
      if ((queued_elapsed >= max_queued_seconds)); then
        echo "WAIT_SETTLED_STATUS=QUEUED_TIMEOUT"
        return 3
      fi
      sleep "$queued_interval"
      queued_elapsed=$((queued_elapsed + queued_interval))
      ;;
    RUNNING)
      queued_elapsed=0
      running_polls=$((running_polls + 1))
      echo "wait: RUNNING poll ${running_polls}/${max_polls}"
      if ((running_polls >= max_polls)); then
        echo "WAIT_SETTLED_STATUS=TIMEOUT_STILL_PENDING"
        return 2
      fi
      sleep "$poll_interval"
      ;;
    "")
      running_polls=$((running_polls + 1))
      printf 'wait: unparseable status (%s/%s); raw output follows\n%s\n' \
        "$running_polls" "$max_polls" "$status_line" >&2
      if ((running_polls >= max_polls)); then
        echo "WAIT_SETTLED_STATUS=TIMEOUT_STILL_PENDING"
        return 2
      fi
      sleep "$poll_interval"
      ;;
    *)
      echo "WAIT_SETTLED_STATUS=${status}"
      return 0
      ;;
    esac
  done
}

require_kaggle_sources_confirmed() {
  local metadata="$1"
  local source_summary
source_summary="$(python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source_fields = (
    "dataset_sources",
    "competition_sources",
    "kernel_sources",
    "model_sources",
)
nonempty = {}
for field in source_fields:
    sources = data.get(field)
    if sources is None:
        continue
    if not isinstance(sources, list):
        print(f"error: {field} must be a list", file=sys.stderr)
        raise SystemExit(2)
    if sources:
        nonempty[field] = sources
if nonempty:
    print(json.dumps(nonempty, sort_keys=True))
PY
)"

  if [[ -n "$source_summary" && "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<EOF
error: $metadata declares Kaggle source attachments: $source_summary

Kaggle source attachments can make Kaggle prepare external datasets, kernels,
models, or competition inputs before the script starts. Set
KAGGLE_FULL_DATASET_CONFIRMED=1 only after explicitly deciding to attach those
sources for this push. Use a no-dataset
synthetic/random benchmark kernel for setup or timing-plumbing tests.
EOF
    exit 1
  fi
}

metadata_path() {
  local kernel_dir="${1:-$default_kernel_dir}"
  printf '%s/kernel-metadata.json\n' "$kernel_dir"
}

json_field() {
  local metadata="$1"
  local field="$2"
  python3 - "$metadata" "$field" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
field = sys.argv[2]
value = json.loads(metadata.read_text(encoding="utf-8")).get(field, "")
if isinstance(value, str):
    print(value)
PY
}

validate_kernel_dir() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  local code_file
  metadata="$(metadata_path "$kernel_dir")"

  if [[ ! -d "$kernel_dir" ]]; then
    echo "missing: $kernel_dir" >&2
    exit 1
  fi

  if [[ ! -f "$metadata" ]]; then
    echo "missing: $metadata" >&2
    exit 1
  fi

  python3 -m json.tool "$metadata" >/dev/null
  code_file="$(json_field "$metadata" code_file)"

  if [[ -z "$code_file" ]]; then
    echo "error: metadata code_file is empty" >&2
    exit 1
  fi

  if [[ ! -f "$kernel_dir/$code_file" ]]; then
    echo "missing: $kernel_dir/$code_file" >&2
    exit 1
  fi

  echo "ok: $kernel_dir"
  echo "ok: $metadata"
  echo "ok: $kernel_dir/$code_file"

  if [[ "$kernel_dir" == "$corrected_rotation_geometry_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_CORRECTED_ROTATION_GEOMETRY_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: Spec 0050 corrected-geometry embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$decoded_transform_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_DECODED_LATENT_TRANSFORM_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: Spec 0051 decoded-transform embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$functional_geometry_preflight_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: Spec 0057 JVP ladder preflight embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$jvp_epsilon_calibration_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_JVP_EPSILON_GRID_CALIBRATION_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: Spec 0058 JVP epsilon calibration embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$real_data_runtime_pretest_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: real-data runtime pretest embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$runtime_selection_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_RUNTIME_SELECTION_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: runtime-selection embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$selected_runtime_debug_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: selected-runtime debug embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$selected_runtime_lr_range_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: selected-runtime LR-range embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$selected_runtime_full_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SELECTED_RUNTIME_FULL_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: selected-runtime full embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$so2_prelaunch_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SO2_PRELAUNCH_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: SO2 prelaunch embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$so2_full_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: SO2 full embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$fixed25_selector_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_FIXED25_SELECTOR_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: fixed25-selector embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$ubc_ocean_test_atlas_kernel_dir" ]]; then
    if ! cmp -s "$ubc_ocean_test_generator" "$kernel_dir/$code_file"; then
      echo "error: atlas run.py does not match $ubc_ocean_test_generator; rebuild it" >&2
      exit 1
    fi
    echo "ok: UBC-OCEAN atlas run.py matches the readable generator"
  fi
}

build_ubc_ocean_test_atlas_kernel() {
  local kernel_dir="${1:-$ubc_ocean_test_atlas_kernel_dir}"
  local metadata
  metadata="$(metadata_path "$kernel_dir")"
  if [[ ! -f "$metadata" ]]; then
    echo "missing: $metadata" >&2
    exit 1
  fi
  if [[ ! -f "$ubc_ocean_test_generator" ]]; then
    echo "missing: $ubc_ocean_test_generator" >&2
    exit 1
  fi
  cp "$ubc_ocean_test_generator" "$kernel_dir/run.py"
  echo "ok: copied $ubc_ocean_test_generator to $kernel_dir/run.py"
  validate_kernel_dir "$kernel_dir"
}

build_kernel_payload() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local payload_dir="$kernel_dir/payload"

  validate_kernel_dir "$kernel_dir"

  if [[ ! -d "src/eqvae" ]]; then
    echo "error: missing src/eqvae; implement spec 0001 before building Kaggle payload" >&2
    exit 1
  fi

  if [[ ! -d "configs/spec0001" ]]; then
    echo "error: missing configs/spec0001; implement spec 0001 before building Kaggle payload" >&2
    exit 1
  fi

  python3 - "$payload_dir" <<'PY'
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

payload = Path(sys.argv[1])
if payload.exists():
    shutil.rmtree(payload)
(payload / "src").mkdir(parents=True)
(payload / "configs").mkdir(parents=True)
ignore_generated = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
shutil.copytree("src/eqvae", payload / "src" / "eqvae", ignore=ignore_generated)
shutil.copytree(
    "configs/spec0001",
    payload / "configs" / "spec0001",
    ignore=ignore_generated,
)
shutil.copy2("pyproject.toml", payload / "pyproject.toml")
shutil.copy2("uv.lock", payload / "uv.lock")


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_tree(path: Path) -> str:
    hasher = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(digest_file(item).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


manifest = {
    "schema_version": "spec0001.kaggle_payload_manifest.v1",
    "git_commit": git_output("rev-parse", "HEAD"),
    "git_dirty": bool(git_output("status", "--short")),
    "entries": {
        "src/eqvae": digest_tree(payload / "src" / "eqvae"),
        "configs/spec0001": digest_tree(payload / "configs" / "spec0001"),
        "pyproject.toml": digest_file(payload / "pyproject.toml"),
        "uv.lock": digest_file(payload / "uv.lock"),
    },
}

(payload / "payload_manifest.json").write_text(
    f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
    encoding="utf-8",
)
PY

  echo "ok: built $payload_dir"
}

is_setup_kernel_dir() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  metadata="$(metadata_path "$kernel_dir")"
  if [[ ! -f "$metadata" ]]; then
    return 1
  fi
  [[ "$(json_field "$metadata" id)" == "maximusshtefan/eqvae-setup-smoke" ]]
}

build_embedded_setup_kernel() {
  local kernel_dir="${1:-$setup_kernel_dir}"
  build_embedded_kernel "$kernel_dir"
}

build_embedded_kernel() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  local ready_marker
  metadata="$(metadata_path "$kernel_dir")"
  ready_marker="$(embedded_ready_marker "$kernel_dir")"

  if [[ ! -f "$metadata" ]]; then
    echo "missing: $metadata" >&2
    exit 1
  fi
  if [[ ! -f "$kernel_dir/run_template.py" ]]; then
    echo "missing: $kernel_dir/run_template.py" >&2
    exit 1
  fi

  python3 -m json.tool "$metadata" >/dev/null
  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "$ready_marker" \
    --allow-dirty
  validate_kernel_dir "$kernel_dir"
}

embedded_ready_marker() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  metadata="$(metadata_path "$kernel_dir")"
  case "$(json_field "$metadata" id)" in
    maximusshtefan/eqvae-setup-smoke)
      printf '%s\n' "KAGGLE_SETUP_SMOKE_READY = True"
      ;;
    maximusshtefan/eqvae-synthetic-timing)
      printf '%s\n' "KAGGLE_SYNTHETIC_TIMING_READY = True"
      ;;
    maximusshtefan/eqvae-real-data-runtime-pretest)
      printf '%s\n' "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True"
      ;;
    maximusshtefan/eqvae-runtime-selection)
      printf '%s\n' "KAGGLE_RUNTIME_SELECTION_READY = True"
      ;;
    maximusshtefan/eqvae-selected-runtime-debug)
      printf '%s\n' "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True"
      ;;
    maximusshtefan/eqvae-selected-runtime-lr-range)
      printf '%s\n' "KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True"
      ;;
    maximusshtefan/eqvae-selected-runtime-full)
      printf '%s\n' "KAGGLE_SELECTED_RUNTIME_FULL_READY = True"
      ;;
    maximusshtefan/eqvae-fixed25-selector)
      printf '%s\n' "KAGGLE_FIXED25_SELECTOR_READY = True"
      ;;
    maximshtefan/eqvae-fixed25-rotation-population)
      printf '%s\n' "KAGGLE_FIXED25_ROTATION_POPULATION_READY = True"
      ;;
    maximshtefan/eqvae-corrected-rotation-geometry)
      printf '%s\n' "KAGGLE_CORRECTED_ROTATION_GEOMETRY_READY = True"
      ;;
    maximshtefan/eqvae-decoded-latent-transform)
      printf '%s\n' "KAGGLE_DECODED_LATENT_TRANSFORM_READY = True"
      ;;
    maximshtefan/eqvae-functional-geometry-preflight-04a08ab5)
      printf '%s\n' "KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_READY = True"
      ;;
    maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5)
      printf '%s\n' "KAGGLE_JVP_EPSILON_GRID_CALIBRATION_READY = True"
      ;;
    maximusshtefan/eqvae-selected-runtime-compile-probe)
      printf '%s\n' "KAGGLE_SELECTED_RUNTIME_COMPILE_PROBE_READY = True"
      ;;
    maximusshtefan/eqvae-so2-architecture-probe)
      printf '%s\n' "KAGGLE_SO2_ARCHITECTURE_PROBE_READY = True"
      ;;
    maximusshtefan/eqvae-so2-runtime-readiness)
      printf '%s\n' "KAGGLE_SO2_RUNTIME_READINESS_READY = True"
      ;;
    maximusshtefan/eqvae-so2-prelaunch)
      printf '%s\n' "KAGGLE_SO2_PRELAUNCH_READY = True"
      ;;
    maximshtefan/eqvae-so2-selected-runtime-full-session7)
      printf '%s\n' "KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True"
      ;;
    maximusshtefan/non-eq-vae-debug)
      printf '%s\n' "KAGGLE_SMOKE_READY = True"
      ;;
    *)
      printf '%s\n' "KAGGLE_SMOKE_READY = True"
      ;;
  esac
}

kernel_id_from_metadata() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  metadata="$(metadata_path "$kernel_dir")"
  json_field "$metadata" id
}

guard_push_ready() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  local code_file
  local kernel_id
  metadata="$(metadata_path "$kernel_dir")"
  code_file="$(json_field "$metadata" code_file)"
  kernel_id="$(json_field "$metadata" id)"

  if [[ "$kernel_dir" == "$functional_geometry_preflight_resume_kernel_dir" \
    || "$kernel_id" == "$functional_geometry_preflight_resume_kernel_id" ]]; then
    if [[ "${KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_RESUME_CONFIRMED:-}" != "1" \
      || "${KAGGLE_REMOTE_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$functional_geometry_preflight_resume_kernel_dir" \
      || "$kernel_id" != "$functional_geometry_preflight_resume_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact receipt-bound Spec 0057 JVP ladder resume confirmation/path required" >&2
      exit 1
    fi
    if [[ -e "$functional_geometry_preflight_resume_claim" ]] \
      || compgen -G \
      "runs/local/kaggle_launches/maximshtefan/eqvae-functional-geometry-preflight-04a08ab5-resume/v*.json" \
      >/dev/null; then
      echo "error: Spec 0057 JVP ladder resume launch authority was already consumed" >&2
      exit 1
    fi
    local resume_actor
    resume_actor="$(kaggle_authenticated_username)"
    if [[ "$resume_actor" != "maximshtefan" ]]; then
      echo "error: Spec 0057 JVP ladder resume launch requires authenticated actor maximshtefan" >&2
      exit 1
    fi
    python3 - "$metadata" "$kernel_dir/continuation_contract.json" \
      "$kernel_dir/$code_file" <<'PYSPEC0057RESUMEGUARD'
import json
import sys
from pathlib import Path

metadata_path, contract_path, source_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expected_metadata = {
    "id": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5-resume",
    "title": "eqvae-functional-geometry-preflight-04a08ab5-resume",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_internet": "false",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": ["maximshtefan/eqvae-functional-geometry-preflight-04a08ab5"],
    "model_sources": [],
}
if metadata != expected_metadata:
    raise SystemExit("Spec 0057 JVP ladder resume metadata differs")
contract = json.loads(contract_path.read_text(encoding="utf-8"))
required = {
    "schema": "spec0057.preflight_continuation_binding.v1",
    "parent_kernel_id": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5",
    "parent_version": 1,
    "parent_reference": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5/1",
    "parent_contract_sha256": "bc48f1f6d4a3088501054aa246cfa2785adec9947914faf76e9b44501e359482",
    "parent_spec_sha256": "6821036a7b616f34ec5ddee339d13bc15b8167b0359e089dbc89a5eed30144b0",
    "pending_work_id": "a8a3d5842f7d9e9f3be45029f6ed55d2e8e6651e2476312c151211259010c472",
    "child_kernel_id": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5-resume",
}
for field, value in required.items():
    if contract.get(field) != value:
        raise SystemExit(f"Spec 0057 JVP ladder resume contract differs at {field}")
manifest_hash = contract.get("parent_manifest_sha256")
if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
    raise SystemExit("Spec 0057 JVP ladder resume contract has no parent manifest hash")
output_receipt_hash = contract.get("parent_output_receipt_sha256")
if not isinstance(output_receipt_hash, str) or len(output_receipt_hash) != 64:
    raise SystemExit("Spec 0057 JVP ladder resume contract has no parent output receipt hash")
source = source_path.read_text(encoding="utf-8")
if source_path.stat().st_size >= 1_000_000:
    raise SystemExit("Spec 0057 JVP ladder resume source exceeds Kaggle's limit")
if json.dumps(contract, sort_keys=True) not in source:
    raise SystemExit("Spec 0057 JVP ladder resume source does not embed the validated contract")
for required_source in (
    "KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_RESUME_READY = True",
    "expected exactly one mounted predecessor",
    "parent_manifest_sha256",
    "preflight_jvp_ladder_resume_v1",
):
    if required_source not in source:
        raise SystemExit(f"Spec 0057 JVP ladder resume source marker missing: {required_source}")
for forbidden in ("torch", "optimizer", "model", "dataset"):
    if forbidden in source.lower():
        raise SystemExit(f"Spec 0057 JVP ladder resume forbidden source token: {forbidden}")
compile(source, str(source_path), "exec")
PYSPEC0057RESUMEGUARD
    return
  fi

  if [[ "$kernel_dir" == "$jvp_epsilon_calibration_kernel_dir" \
    || "$kernel_id" == "$jvp_epsilon_calibration_kernel_id" ]]; then
    if [[ "${KAGGLE_JVP_EPSILON_GRID_CALIBRATION_CONFIRMED:-}" != "1" \
      || "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$jvp_epsilon_calibration_kernel_dir" \
      || "$kernel_id" != "$jvp_epsilon_calibration_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0058 JVP calibration confirmation/path required" >&2
      exit 1
    fi
    if ! grep -q 'spec0058_jvp_epsilon_grid_calibration_authorized' \
      docs/specs/0058-jvp-epsilon-grid-calibration.md \
      || ! grep -q 'spec0058_jvp_epsilon_grid_calibration_authorized' \
      docs/specs/README.md; then
      echo "error: Spec 0058 calibration authorization is not canonical" >&2
      exit 1
    fi
    if [[ -e "$jvp_epsilon_calibration_claim" ]] \
      || compgen -G \
      "runs/local/kaggle_launches/maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5/v*.json" \
      >/dev/null; then
      echo "error: Spec 0058 calibration launch authority was already consumed" >&2
      exit 1
    fi
    local calibration_actor
    calibration_actor="$(kaggle_authenticated_username)"
    if [[ "$calibration_actor" != "maximshtefan" ]]; then
      echo "error: Spec 0058 calibration launch requires actor maximshtefan" >&2
      exit 1
    fi
    python3 - "$metadata" "$kernel_dir/$code_file" <<'PYSPEC0058GUARD'
import json
import sys
from pathlib import Path

metadata_path, source_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expected = {
    "id": "maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5",
    "title": "eqvae-jvp-epsilon-grid-calibration-05a08ab5",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "maximshtefan/eqvae-vae-test-reconstruction-inputs-v1",
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
if metadata != expected:
    raise SystemExit("Spec 0058 calibration metadata differs")
source = source_path.read_text(encoding="utf-8")
if source_path.stat().st_size >= 1_000_000:
    raise SystemExit("Spec 0058 calibration source exceeds Kaggle's limit")
required = (
    "KAGGLE_JVP_EPSILON_GRID_CALIBRATION_READY = True",
    'CONTRACT_SHA256 = "397b2cd6efc4bb7e4b3d775b27e58c63d6d8a7f4f2c0a68fb600ba337d183e08"',
    'SPEC_SHA256 = "037e6f74b3b0f06b8fab3b43da62a6505bbfdd51d98be8d913a04fa3836223fb"',
    'OUTPUT_ROOT = WORKING_ROOT / "jvp_epsilon_grid_calibration_v1"',
    "frozen_bundle_ready",
    "torch.autograd.functional.jvp",
    "EPSILON_GRID = (0.002, 0.004, 0.008, 0.016)",
    "SELECTION_MAX = 0.005",
    "branch_a",
    "branch_b",
)
missing = [marker for marker in required if marker not in source]
if missing:
    raise SystemExit(f"Spec 0058 calibration source markers missing: {missing}")
for forbidden in ("torch.optim", ".backward(", "torch.compile(", "functional.vjp"):
    if forbidden in source.lower():
        raise SystemExit(f"Spec 0058 calibration forbidden source token: {forbidden}")
compile(source, str(source_path), "exec")
PYSPEC0058GUARD
    return
  fi

  if [[ "$kernel_dir" == "$functional_geometry_preflight_kernel_dir" \
    || "$kernel_id" == "$functional_geometry_preflight_kernel_id" ]]; then
    if [[ "${KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$functional_geometry_preflight_kernel_dir" \
      || "$kernel_id" != "$functional_geometry_preflight_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0057 JVP ladder confirmation/path required" >&2
      exit 1
    fi
    if ! grep -q 'spec0057_jvp_epsilon_ladder_authorized' \
      docs/specs/0057-jvp-epsilon-ladder-preflight.md \
      || ! grep -q 'spec0057_jvp_epsilon_ladder_authorized' \
      docs/specs/README.md; then
      echo "error: Spec 0057 JVP ladder authorization is not canonical" >&2
      exit 1
    fi
    if [[ -e "$functional_geometry_preflight_claim" ]] \
      || compgen -G \
      "runs/local/kaggle_launches/maximshtefan/eqvae-functional-geometry-preflight-04a08ab5/v*.json" \
      >/dev/null; then
      echo "error: Spec 0057 JVP ladder launch authority was already consumed" >&2
      exit 1
    fi
    local preflight_actor
    preflight_actor="$(kaggle_authenticated_username)"
    if [[ "$preflight_actor" != "maximshtefan" ]]; then
      echo "error: Spec 0057 JVP ladder launch requires authenticated actor maximshtefan" >&2
      exit 1
    fi
    python3 - "$metadata" "$kernel_dir/$code_file" <<'PYSPEC0053PREFLIGHT'
import json
import sys
from pathlib import Path

metadata_path, source_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expected = {
    "id": "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5",
    "title": "eqvae-functional-geometry-preflight-04a08ab5",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "maximshtefan/eqvae-vae-test-reconstruction-inputs-v1",
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
if metadata != expected:
    raise SystemExit("Spec 0057 JVP ladder metadata differs")
source = source_path.read_text(encoding="utf-8")
if source_path.stat().st_size >= 1_000_000:
    raise SystemExit("Spec 0057 JVP ladder source exceeds Kaggle's limit")
required = (
    "KAGGLE_FUNCTIONAL_GEOMETRY_JVP_LADDER_READY = True",
    'CONTRACT_SHA256 = "bc48f1f6d4a3088501054aa246cfa2785adec9947914faf76e9b44501e359482"',
    'SPEC_SHA256 = "6821036a7b616f34ec5ddee339d13bc15b8167b0359e089dbc89a5eed30144b0"',
    'OUTPUT_ROOT = WORKING_ROOT / "preflight_jvp_ladder_parent_v1"',
    "frozen_bundle_ready",
    '"jvp_diagnostic_epsilons"',
    "jvp_epsilon_0_004",
    "jvp_epsilon_0_008",
    "b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1",
    "branch_a",
    "branch_b",
    "a CUDA GPU is required",
)
missing = [marker for marker in required if marker not in source]
if missing:
    raise SystemExit(f"Spec 0057 JVP ladder source markers missing: {missing}")
for forbidden in ("torch.optim", ".backward("):
    if forbidden in source.lower():
        raise SystemExit(f"Spec 0057 JVP ladder forbidden source token: {forbidden}")
compile(source, str(source_path), "exec")
PYSPEC0053PREFLIGHT
    return
  fi

  if [[ "$kernel_dir" == "$decoded_transform_kernel_dir" \
    || "$kernel_id" == "$decoded_transform_kernel_id" ]]; then
    if [[ "${KAGGLE_DECODED_LATENT_TRANSFORM_CONFIRMED:-}" != "1" \
      || "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$decoded_transform_kernel_dir" \
      || "$kernel_id" != "$decoded_transform_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0051 one-shot confirmation/path required" >&2
      exit 1
    fi
    if ! grep -q 'spec0051_decoded_latent_transform_authorized' \
      docs/specs/0051-decoded-latent-transform-consistency.md \
      || ! grep -q 'spec0051_decoded_latent_transform_authorized' \
      docs/specs/README.md; then
      echo "error: Spec 0051 remote authorization is not canonical" >&2
      exit 1
    fi
    if [[ -e "$decoded_transform_claim" ]] \
      || compgen -G \
      "runs/local/kaggle_launches/maximshtefan/eqvae-decoded-latent-transform/v*.json" \
      >/dev/null; then
      echo "error: Spec 0051 one-shot launch authority was already consumed" >&2
      exit 1
    fi
    local decoded_actor
    decoded_actor="$(kaggle_authenticated_username)"
    if [[ "$decoded_actor" != "maximshtefan" ]]; then
      echo "error: Spec 0051 launch requires authenticated actor maximshtefan" >&2
      exit 1
    fi
    python3 - "$metadata" "$kernel_dir/$code_file" <<'PYSPEC0051TRANSFORM'
import json
import sys
from pathlib import Path

metadata_path, source_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expected = {
    "id": "maximshtefan/eqvae-decoded-latent-transform",
    "title": "EQVAE decoded latent transform audit",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "maximshtefan/eqvae-vae-test-reconstruction-inputs-v1",
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
if metadata != expected:
    raise SystemExit("Spec 0051 metadata differs")
source = source_path.read_text(encoding="utf-8")
if source_path.stat().st_size >= 1_000_000:
    raise SystemExit("Spec 0051 source exceeds Kaggle's limit")
required = (
    "KAGGLE_DECODED_LATENT_TRANSFORM_READY = True",
    'CONTRACT_SHA256 = "905ef933a51c26bb3f01fcef4bb4985fa98e00f1dbecd814da3df488bd7c1018"',
    'SPEC_SHA256 = "7a472a2de56813544506b4a9eb58e2f1f152fcee8f7d60e65ed2c2793b349679"',
    "range(0, 360, 5)",
    "EXACT_D4_NONIDENTITY_NAMES",
    "a CUDA GPU is required",
    'OUTPUT_ROOT = WORKING_ROOT / "decoded_latent_transform_v1"',
)
missing = [marker for marker in required if marker not in source]
if missing:
    raise SystemExit(f"Spec 0051 source contract markers missing: {missing}")
for forbidden in ("optimizer", ".backward(", ".step("):
    if forbidden in source.lower():
        raise SystemExit(f"Spec 0051 must remain inference-only: {forbidden}")
compile(source, str(source_path), "exec")
PYSPEC0051TRANSFORM
    return
  fi

  if [[ "$kernel_dir" == "$corrected_rotation_geometry_kernel_dir" \
    || "$kernel_id" == "$corrected_rotation_geometry_kernel_id" ]]; then
    if [[ "${KAGGLE_CORRECTED_ROTATION_GEOMETRY_CONFIRMED:-}" != "1" \
      || "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$corrected_rotation_geometry_kernel_dir" \
      || "$kernel_id" != "$corrected_rotation_geometry_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0050 one-shot confirmation/path required" >&2
      exit 1
    fi
    if ! grep -q 'spec0050_corrected_rotation_geometry_authorized' \
      docs/specs/0050-corrected-rotation-geometry-validation.md \
      || ! grep -q 'spec0050_corrected_rotation_geometry_authorized' \
      docs/specs/README.md; then
      echo "error: Spec 0050 remote authorization is not canonical" >&2
      exit 1
    fi
    if [[ -e "$corrected_rotation_geometry_claim" ]] \
      || compgen -G \
      "runs/local/kaggle_launches/maximshtefan/eqvae-corrected-rotation-geometry/v*.json" \
      >/dev/null; then
      echo "error: Spec 0050 one-shot launch authority was already consumed" >&2
      exit 1
    fi
    local rotation_actor
    rotation_actor="$(kaggle_authenticated_username)"
    if [[ "$rotation_actor" != "maximshtefan" ]]; then
      echo "error: Spec 0050 launch requires authenticated actor maximshtefan" >&2
      exit 1
    fi
    python3 - "$metadata" "$kernel_dir/$code_file" <<'PYSPEC0050GEOMETRY'
import json
import sys
from pathlib import Path

metadata_path, source_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expected = {
    "id": "maximshtefan/eqvae-corrected-rotation-geometry",
    "title": "EQVAE corrected rotation geometry",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "maximshtefan/eqvae-vae-test-reconstruction-inputs-v1",
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
if metadata != expected:
    raise SystemExit("Spec 0050 metadata differs")
source = source_path.read_text(encoding="utf-8")
if source_path.stat().st_size >= 1_000_000:
    raise SystemExit("Spec 0050 source exceeds Kaggle's limit")
required = (
    "KAGGLE_CORRECTED_ROTATION_GEOMETRY_READY = True",
    'CONTRACT_SHA256 = "1dc6975979b120d85680526a3177e30caf93ca2eba7889a4452f2e78ca399756"',
    'SPEC_SHA256 = "fbc58459d214ee32f1148f6b58507820586be1993749811294f14eddbf467b61"',
    "range(360)",
    "a CUDA GPU is required",
    'OUTPUT_ROOT = WORKING_ROOT / "corrected_rotation_geometry_v1"',
)
missing = [marker for marker in required if marker not in source]
if missing:
    raise SystemExit(f"Spec 0050 source contract markers missing: {missing}")
for forbidden in ("optimizer", ".backward(", ".step("):
    if forbidden in source.lower():
        raise SystemExit(f"Spec 0050 must remain inference-only: {forbidden}")
compile(source, str(source_path), "exec")
PYSPEC0050GEOMETRY
    return
  fi

  if [[ "$kernel_dir" == "$fixed25_rotation_population_kernel_dir" \
    || "$kernel_id" == "$fixed25_rotation_population_kernel_id" ]]; then
    if [[ "${KAGGLE_FIXED25_ROTATION_POPULATION_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$fixed25_rotation_population_kernel_dir" \
      || "$kernel_id" != "$fixed25_rotation_population_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0038 dense-360 confirmation/path required" >&2
      exit 1
    fi
    if ! grep -q 'spec0038_fixed25_dense360_population_authorized' \
      docs/specs/0038-frozen-vae-rotation-orbit-visualization.md \
      || ! grep -q 'spec0038_fixed25_dense360_population_authorized' \
      docs/specs/README.md; then
      echo "error: Spec 0038 dense-360 remote authorization is not canonical" >&2
      exit 1
    fi
    if [[ -f \
      "runs/local/kaggle_launches/maximshtefan/eqvae-fixed25-rotation-population/v0001.json" \
      || -f \
      "runs/local/kaggle_launches/maximshtefan/eqvae-fixed25-dense-rotation-population/v0001.json" ]]; then
      echo "error: Spec 0038 dense-360 launch authority was already consumed" >&2
      exit 1
    fi
    python3 - "$metadata" "$kernel_dir/$code_file" <<'PYSPEC0038DENSE360'
import json
import sys
from pathlib import Path

metadata_path, source_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expected = {
    "id": "maximshtefan/eqvae-fixed25-rotation-population",
    "title": "EQVAE fixed25 dense rotation population",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "maximshtefan/eqvae-vae-test-reconstruction-inputs-v1",
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
if metadata != expected:
    raise SystemExit("Spec 0038 dense-360 metadata differs")
source = source_path.read_text(encoding="utf-8")
if source_path.stat().st_size >= 1_000_000:
    raise SystemExit("Spec 0038 dense-360 source exceeds Kaggle's limit")
if "KAGGLE_FIXED25_ROTATION_POPULATION_READY = True" not in source:
    raise SystemExit("Spec 0038 dense-360 ready marker is absent")
if "range(360)" not in source or "a CUDA GPU is required" not in source:
    raise SystemExit("Spec 0038 dense-360 angle or CUDA contract differs")
if "optimizer" in source.lower():
    raise SystemExit("Spec 0038 dense-360 must remain inference-only")
compile(source, str(source_path), "exec")
PYSPEC0038DENSE360
    return
  fi

  if [[ "$kernel_dir" == "$vae_test_kernel_dir" \
    || "$kernel_id" == "maximshtefan/$vae_test_kernel_slug" ]]; then
    if [[ "${KAGGLE_VAE_TEST_ROUTE_ACTIVE:-}" != "1" \
      || "${KAGGLE_VAE_TEST_EVALUATION_CONFIRMED:-}" != "1" \
      || "${EQVAE_KAGGLE_LAUNCH_RECEIPT_ROOT:-runs/local/kaggle_launches}" \
        != "runs/local/kaggle_launches" \
      || "$kernel_dir" != "$vae_test_kernel_dir" \
      || "$kernel_id" != "maximshtefan/$vae_test_kernel_slug" \
      || "$code_file" != "run.py" ]]; then
      echo "error: use the exact one-shot push-vae-test route for Spec 0045" >&2
      exit 1
    fi
    local vae_test_actor
    vae_test_actor="$(kaggle_authenticated_username)"
    [[ "$kernel_id" == "$vae_test_actor/$vae_test_kernel_slug" ]] || {
      echo "error: Spec 0045 authenticated actor/kernel identity differs" >&2
      exit 1
    }
    require_build_python
    "$build_python" scripts/build_vae_test_evaluation.py \
      validate-claimed-launch --actor "$vae_test_actor" >/dev/null
    return
  fi

  if [[ "$kernel_dir" == "$tissue_test_kernel_dir" \
    || "$kernel_id" == "maximshtefan/$tissue_test_kernel_slug" ]]; then
    if [[ "${KAGGLE_TISSUE_TEST_ROUTE_ACTIVE:-}" != "1" \
      || "${KAGGLE_TISSUE_TEST_EVALUATION_CONFIRMED:-}" != "1" \
      || "${EQVAE_KAGGLE_LAUNCH_RECEIPT_ROOT:-runs/local/kaggle_launches}" \
        != "runs/local/kaggle_launches" \
      || "$kernel_dir" != "$tissue_test_kernel_dir" \
      || "$kernel_id" != "maximshtefan/$tissue_test_kernel_slug" \
      || "$code_file" != "run.py" ]]; then
      echo "error: use the exact one-shot push-tissue-test route for Spec 0043" >&2
      exit 1
    fi
    local tissue_test_actor
    tissue_test_actor="$(kaggle_authenticated_username)"
    [[ "$kernel_id" == "$tissue_test_actor/$tissue_test_kernel_slug" ]] || {
      echo "error: Spec 0043 authenticated actor/kernel identity differs" >&2
      exit 1
    }
    require_build_python
    "$build_python" scripts/build_tissue_test_evaluation.py \
      validate-claimed-launch --actor "$tissue_test_actor" >/dev/null
    return
  fi

  # Spec 0031 has no remote authority yet. Keep both immutable identities out
  # of the generic uploader until a separately reviewed one-shot route exists.
  if [[ "$kernel_dir" == "$flex_attention_probe_kernel_dir" \
    || "$kernel_id" == "$flex_attention_probe_kernel_id" ]]; then
    echo "error: Spec 0031 FlexAttention remote launch is not authorized" >&2
    exit 1
  fi

  if [[ "$kernel_dir" == "$inductor_attention_probe_kernel_dir" \
    || "$kernel_id" == "$inductor_attention_probe_kernel_id" ]]; then
    if [[ "${KAGGLE_INDUCTOR_ATTENTION_PROBE_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$inductor_attention_probe_kernel_dir" \
      || "$kernel_id" != "$inductor_attention_probe_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0033 Inductor-attention probe confirmation/path required" >&2
      exit 1
    fi
    if ! grep -q 'spec0033_inductor_attention_probe_authorized' \
      docs/specs/0033-exact-local-attention-backend-bakeoff.md \
      || ! grep -q 'spec0033_inductor_attention_probe_authorized' \
      docs/specs/README.md; then
      echo "error: Spec 0033 remote authorization is not canonical" >&2
      exit 1
    fi
    python3 - "$metadata" "$kernel_dir/$code_file" <<'PYINDUCTORATTENTION'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = Path(sys.argv[2])
assert metadata == {
    "id": "maximusshtefan/eqvae-wsi45630-inductor-attention-probe",
    "title": "eqvae WSI45630 Inductor attention probe",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["maximusshtefan/eqvae-wsi45630-capacity-inputs"],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
code = source.read_text(encoding="utf-8")
assert source.stat().st_size < 1_000_000
assert "SPEC0033_INDUCTOR_ATTENTION_PROBE_READY = True" in code
assert "INPUT_CONTRACT_SHA256" in code and "POINTER_SHA256" in code
assert "optimizer" not in code.lower()
compile(code, str(source), "exec")
PYINDUCTORATTENTION
    return
  fi

  if [[ "$kernel_dir" == "$full_compile_probe_kernel_dir" \
    || "$kernel_id" == "$full_compile_probe_kernel_id" ]]; then
    if [[ "${KAGGLE_FULL_COMPILE_PROBE_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$full_compile_probe_kernel_dir" \
      || "$kernel_id" != "$full_compile_probe_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0034 full-compile probe confirmation/path required" >&2
      exit 1
    fi
    if [[ ! -f \
      "runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0001.json" ]]; then
      echo "error: Spec 0034 version-1 launch receipt is required" >&2
      exit 1
    fi
    if [[ ! -f \
      "runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0002.json" ]]; then
      echo "error: Spec 0034 version-2 launch receipt is required" >&2
      exit 1
    fi
    if [[ ! -f \
      "runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0003.json" ]]; then
      echo "error: Spec 0034 version-3 launch receipt is required" >&2
      exit 1
    fi
    if [[ ! -f \
      "runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0004.json" ]]; then
      echo "error: Spec 0034 version-4 launch receipt is required" >&2
      exit 1
    fi
    if [[ ! -f \
      "runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0005.json" ]]; then
      echo "error: Spec 0034 version-5 launch receipt is required" >&2
      exit 1
    fi
    if [[ ! -f \
      "runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0006.json" ]]; then
      echo "error: Spec 0034 version-6 launch receipt is required" >&2
      exit 1
    fi
    if [[ -f \
      "runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0007.json" ]]; then
      echo "error: Spec 0034 version-7 retry authority was already consumed" >&2
      exit 1
    fi
    if ! grep -q 'spec0034_pinned_torch_retry_v7_authorized' \
      docs/specs/0034-full-compiled-fixed25-mil-probe.md \
      || ! grep -q 'spec0034_pinned_torch_retry_v7_authorized' \
      docs/specs/README.md; then
      echo "error: Spec 0034 remote authorization is not canonical" >&2
      exit 1
    fi
    require_build_python
    "$build_python" scripts/build_wsi45630_full_compile_probe.py validate >/dev/null
    return
  fi

  if [[ "$kernel_dir" == "$tissue_training_retry_v3_kernel_dir" ]]; then
    if [[ "${KAGGLE_TISSUE_TRAINING_CONFIRMED:-}" != "1" \
      || "${KAGGLE_TISSUE_TRAINING_RETRY_V3_CONFIRMED:-}" != "1" \
      || "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" \
      || "$kernel_id" != "$tissue_training_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0039 retry-v3 confirmation/path required" >&2
      exit 1
    fi
    local actor expected_reference
    actor="$(kaggle_authenticated_username)"
    expected_reference="$actor/$tissue_training_dataset_slug"
    validate_tissue_training_retry_v3 "$actor"
    "$build_python" - \
      "$tissue_training_input_receipt" \
      "$tissue_training_retry_v3_root/bundle/tissue_training_input.json" \
      "runs/local/kaggle_launches/$actor/eqvae-tissue-label-efficiency-training/v0002.json" \
      "runs/kaggle/tissue_label_efficiency_training_v0002/kaggle_output_receipt.json" \
      "$expected_reference" <<'PYSPEC0039RETRYV3PUSH'
import hashlib
import json
import sys
from pathlib import Path

input_receipt_path, contract_path, prior_launch_path, output_receipt_path, reference = (
    map(Path, sys.argv[1:])
)
reference = str(reference)
if not all(
    path.is_file()
    for path in (input_receipt_path, prior_launch_path, output_receipt_path)
):
    raise SystemExit("Spec 0039 retry-v3 requires verified v1 input and v2 output receipts")
input_receipt = json.loads(input_receipt_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
prior_launch = json.loads(prior_launch_path.read_text(encoding="utf-8"))
output_receipt = json.loads(output_receipt_path.read_text(encoding="utf-8"))
kernel_id = reference.replace("-inputs", "")
overall_name = "tissue_label_efficiency_training/spec0039_tissue_training.json"
if (
    input_receipt.get("schema_version") != "spec0039.input_dataset_receipt.v1"
    or input_receipt.get("dataset_reference") != reference
    or input_receipt.get("dataset_version") != 1
    or input_receipt.get("visibility") != "private"
    or input_receipt.get("status") != "verified"
    or input_receipt.get("input_contract_sha256")
    != hashlib.sha256(contract_path.read_bytes()).hexdigest()
    or contract.get("dataset_reference") != reference
    or prior_launch.get("schema_version") != "eqvae.kaggle_kernel_launch.v1"
    or prior_launch.get("accepted_version") != 2
    or prior_launch.get("kernel_reference") != f"{kernel_id}/2"
    or prior_launch.get("source_locators", {}).get("dataset_sources") != [reference]
    or output_receipt.get("schema_version") != "eqvae.kaggle_download.v1"
    or output_receipt.get("resource_kind") != "kernel"
    or output_receipt.get("resource_reference") != f"{kernel_id}/2"
    or overall_name not in output_receipt.get("files", {})
):
    raise SystemExit("Spec 0039 retry-v3 v1-input or v2-result binding differs")
PYSPEC0039RETRYV3PUSH
    if [[ -e "$tissue_training_retry_v3_launch_claim" ]]; then
      echo "error: Spec 0039 retry-v3 launch authority is consumed" >&2
      exit 1
    fi
    mkdir -p "$tissue_training_retry_v3_authority_root"
    "$build_python" - \
      "$tissue_training_retry_v3_launch_claim" \
      "$tissue_training_retry_v3_root/bundle/tissue_training_input.json" \
      "$actor" <<'PYSPEC0039RETRYV3CLAIM'
import hashlib
import json
import os
import sys
from pathlib import Path

claim_path, contract_path, actor = map(Path, sys.argv[1:])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
claim = {
    "schema_version": "spec0039.kernel_launch_retry_v3_claim.v1",
    "status": "claimed_before_remote_push",
    "actor": str(actor),
    "input_dataset_reference": contract["dataset_reference"],
    "input_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "prior_kernel_reference": f"{actor}/eqvae-tissue-label-efficiency-training/2",
    "minimum_completed_epochs": 10,
}
try:
    with claim_path.open("x", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
except FileExistsError as error:
    raise SystemExit("Spec 0039 retry-v3 launch claim already exists") from error
directory = os.open(claim_path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0039RETRYV3CLAIM
    if ! grep -q '^SPEC0039_TISSUE_TRAINING_READY = True$' \
      "$kernel_dir/$code_file" \
      || ! grep -q '^DEFAULT_MINIMUM_COMPLETED_EPOCHS = 10$' \
      "$kernel_dir/$code_file"; then
      echo "error: Spec 0039 retry-v3 minimum-epoch launcher differs" >&2
      exit 1
    fi
    return
  fi

  if [[ "$kernel_dir" == "$tissue_training_retry_kernel_dir" ]]; then
    if [[ "${KAGGLE_TISSUE_TRAINING_CONFIRMED:-}" != "1" \
      || "${KAGGLE_TISSUE_TRAINING_RETRY_V2_CONFIRMED:-}" != "1" \
      || "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" \
      || "$kernel_id" != "$tissue_training_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0039 retry-v2 confirmation/path required" >&2
      exit 1
    fi
    local actor expected_reference
    actor="$(kaggle_authenticated_username)"
    expected_reference="$actor/$tissue_training_dataset_slug"
    validate_tissue_training_retry "$actor"
    "$build_python" - \
      "$tissue_training_input_receipt" \
      "$tissue_training_retry_root/bundle/tissue_training_input.json" \
      "runs/local/kaggle_launches/$actor/eqvae-tissue-label-efficiency-training/v0001.json" \
      "$expected_reference" <<'PYSPEC0039RETRYPUSH'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, contract_path, failed_receipt_path, reference = map(Path, sys.argv[1:])
reference = str(reference)
if not receipt_path.is_file() or not failed_receipt_path.is_file():
    raise SystemExit("Spec 0039 retry requires verified v1 input and failed-v1 receipts")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
failed = json.loads(failed_receipt_path.read_text(encoding="utf-8"))
kernel_id = reference.replace("-inputs", "")
if (
    receipt.get("schema_version") != "spec0039.input_dataset_receipt.v1"
    or receipt.get("dataset_reference") != reference
    or receipt.get("dataset_version") != 1
    or receipt.get("visibility") != "private"
    or receipt.get("status") != "verified"
    or receipt.get("input_contract_sha256")
    != hashlib.sha256(contract_path.read_bytes()).hexdigest()
    or contract.get("dataset_reference") != reference
    or failed.get("schema_version") != "eqvae.kaggle_kernel_launch.v1"
    or failed.get("accepted_version") != 1
    or failed.get("kernel_reference") != f"{kernel_id}/1"
    or failed.get("source_locators", {}).get("dataset_sources") != [reference]
):
    raise SystemExit("Spec 0039 retry-v2 frozen input or failed-v1 binding differs")
PYSPEC0039RETRYPUSH
    if [[ -e "$tissue_training_retry_launch_claim" ]]; then
      echo "error: Spec 0039 retry-v2 launch authority is consumed" >&2
      exit 1
    fi
    mkdir -p "$tissue_training_retry_authority_root"
    "$build_python" - \
      "$tissue_training_retry_launch_claim" \
      "$tissue_training_retry_root/bundle/tissue_training_input.json" \
      "$actor" <<'PYSPEC0039RETRYCLAIM'
import hashlib
import json
import os
import sys
from pathlib import Path

claim_path, contract_path, actor = map(Path, sys.argv[1:])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
claim = {
    "schema_version": "spec0039.kernel_launch_retry_v2_claim.v1",
    "status": "claimed_before_remote_push",
    "actor": str(actor),
    "input_dataset_reference": contract["dataset_reference"],
    "input_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "failed_kernel_reference": f"{actor}/eqvae-tissue-label-efficiency-training/1",
}
try:
    with claim_path.open("x", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
except FileExistsError as error:
    raise SystemExit("Spec 0039 retry-v2 launch claim already exists") from error
directory = os.open(claim_path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0039RETRYCLAIM
    if ! grep -q '^SPEC0039_TISSUE_TRAINING_READY = True$' \
      "$kernel_dir/$code_file"; then
      echo "error: Spec 0039 retry-v2 readiness marker differs" >&2
      exit 1
    fi
    return
  fi

  if [[ "$kernel_dir" == "$tissue_training_kernel_dir" \
    || "$kernel_id" == "$tissue_training_kernel_id" ]]; then
    if [[ "${KAGGLE_TISSUE_TRAINING_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$tissue_training_kernel_dir" \
      || "$kernel_id" != "$tissue_training_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0039 tissue-training confirmation/path required" >&2
      exit 1
    fi
    if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
      echo "error: set KAGGLE_FULL_DATASET_CONFIRMED=1 before Spec 0039's one launch" >&2
      exit 1
    fi
    local actor expected_reference
    actor="$(kaggle_authenticated_username)"
    expected_reference="$actor/$tissue_training_dataset_slug"
    validate_tissue_training "$actor" sealed
    "$build_python" - \
      "$tissue_training_input_receipt" \
      "$tissue_training_root/bundle/tissue_training_input.json" \
      "$expected_reference" <<'PYSPEC0039PUSH'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, contract_path, reference = map(Path, sys.argv[1:])
reference = str(reference)
if not receipt_path.is_file():
    raise SystemExit("Spec 0039 push requires its verified input receipt")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if (
    receipt.get("schema_version") != "spec0039.input_dataset_receipt.v1"
    or receipt.get("dataset_reference") != reference
    or receipt.get("dataset_version") != 1
    or receipt.get("visibility") != "private"
    or receipt.get("status") != "verified"
    or receipt.get("input_contract_sha256")
    != hashlib.sha256(contract_path.read_bytes()).hexdigest()
    or contract.get("dataset_reference") != reference
):
    raise SystemExit("Spec 0039 verified input receipt binding differs")
PYSPEC0039PUSH
    if [[ -e "$tissue_training_launch_claim" ]]; then
      echo "error: Spec 0039's one private kernel-launch authority is consumed" >&2
      exit 1
    fi
    mkdir -p "$tissue_training_authority_root"
    "$build_python" - \
      "$tissue_training_launch_claim" \
      "$tissue_training_root/bundle/tissue_training_input.json" \
      "$actor" <<'PYSPEC0039CLAIM'
import hashlib
import json
import os
import sys
from pathlib import Path

claim_path, contract_path, actor = map(Path, sys.argv[1:])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
claim = {
    "schema_version": "spec0039.kernel_launch_claim.v1",
    "status": "claimed_before_remote_push",
    "actor": str(actor),
    "input_dataset_reference": contract["dataset_reference"],
    "input_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
}
try:
    with claim_path.open("x", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
except FileExistsError as error:
    raise SystemExit("Spec 0039 launch claim already exists") from error
directory = os.open(claim_path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0039CLAIM
    if ! grep -q '^SPEC0039_TISSUE_TRAINING_READY = True$' \
      "$kernel_dir/$code_file"; then
      echo "error: Spec 0039 tissue-training readiness marker differs" >&2
      exit 1
    fi
    return
  fi

  if [[ "$kernel_dir" == "$tissue_fastpath_probe_kernel_dir" \
    || "$kernel_id" == "$tissue_fastpath_probe_kernel_id" ]]; then
    if [[ "${KAGGLE_TISSUE_FASTPATH_PROBE_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$tissue_fastpath_probe_kernel_dir" \
      || "$kernel_id" != "$tissue_fastpath_probe_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0037 tissue probe confirmation/path required" >&2
      exit 1
    fi
    if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
      echo "error: set KAGGLE_FULL_DATASET_CONFIRMED=1 before claiming Spec 0037's one launch" >&2
      exit 1
    fi
    local actor expected_reference
    actor="$(kaggle_authenticated_username)"
    expected_reference="$actor/$tissue_fastpath_probe_dataset_slug"
    validate_tissue_fastpath_probe "$actor" sealed
    "$build_python" - \
      "$tissue_fastpath_probe_input_receipt" \
      "$tissue_fastpath_probe_root/bundle/tissue_fastpath_probe_input.json" \
      "$expected_reference" <<'PYSPEC0037PUSH'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, contract_path, reference = map(Path, sys.argv[1:])
reference = str(reference)
if not receipt_path.is_file():
    raise SystemExit("Spec 0037 push requires its verified input receipt")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if (
    receipt.get("schema_version") != "spec0037.input_dataset_receipt.v1"
    or receipt.get("dataset_reference") != reference
    or receipt.get("dataset_version") != 1
    or receipt.get("visibility") != "private"
    or receipt.get("status") != "verified"
    or receipt.get("input_contract_sha256")
    != hashlib.sha256(contract_path.read_bytes()).hexdigest()
    or contract.get("dataset_reference") != reference
):
    raise SystemExit("Spec 0037 verified input receipt binding differs")
PYSPEC0037PUSH
    if [[ -e "$tissue_fastpath_probe_launch_claim" ]]; then
      echo "error: Spec 0037's one private kernel-launch authority is consumed" >&2
      exit 1
    fi
    mkdir -p "$tissue_fastpath_probe_authority_root"
    "$build_python" - \
      "$tissue_fastpath_probe_launch_claim" \
      "$tissue_fastpath_probe_root/bundle/tissue_fastpath_probe_input.json" \
      "$actor" <<'PYSPEC0037CLAIM'
import hashlib
import json
import os
import sys
from pathlib import Path

claim_path, contract_path, actor = map(Path, sys.argv[1:])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
claim = {
    "schema_version": "spec0037.kernel_launch_claim.v1",
    "status": "claimed_before_remote_push",
    "actor": str(actor),
    "input_dataset_reference": contract["dataset_reference"],
    "input_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
}
try:
    with claim_path.open("x", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
except FileExistsError as error:
    raise SystemExit("Spec 0037 launch claim already exists") from error
directory = os.open(claim_path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0037CLAIM
    if ! grep -q '^SPEC0037_TISSUE_FASTPATH_PROBE_READY = True$' \
      "$kernel_dir/$code_file"; then
      echo "error: Spec 0037 tissue-probe readiness marker differs" >&2
      exit 1
    fi
    return
  fi

  if [[ "$kernel_dir" == "$mil_test_kernel_dir" \
    || "$kernel_id" == */"$mil_test_kernel_slug" ]]; then
    if [[ "${KAGGLE_MIL_TEST_EVALUATION_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$mil_test_kernel_dir" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0041 MIL test confirmation/path required" >&2
      exit 1
    fi
    local actor expected_kernel_id
    actor="$(kaggle_authenticated_username)"
    expected_kernel_id="$actor/$mil_test_kernel_slug"
    if [[ "$kernel_id" != "$expected_kernel_id" ]]; then
      echo "error: Spec 0041 kernel actor differs" >&2
      exit 1
    fi
    validate_mil_test "$actor" >/dev/null
    "$build_python" - "$mil_test_input_receipt" \
      "$mil_test_root/bundle/mil_test_inference_input.json" \
      "$actor/$mil_test_dataset_slug" <<'PYSPEC0041PUSH'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, contract_path, reference = map(Path, sys.argv[1:])
if not receipt_path.is_file():
    raise SystemExit("Spec 0041 push requires its verified input receipt")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if (
    receipt.get("schema_version") != "spec0041.input_dataset_receipt.v1"
    or receipt.get("dataset_reference") != str(reference)
    or receipt.get("dataset_version") != 1
    or receipt.get("visibility") != "private"
    or receipt.get("status") != "verified"
    or receipt.get("input_contract_sha256")
    != hashlib.sha256(contract_path.read_bytes()).hexdigest()
):
    raise SystemExit("Spec 0041 verified input receipt differs")
PYSPEC0041PUSH
    "$build_python" - "$mil_test_launch_claim" \
      "$mil_test_root/bundle/mil_test_inference_input.json" \
      "$mil_test_kernel_dir/run.py" \
      "$mil_test_kernel_dir/kernel-metadata.json" <<'PYSPEC0041CLAIM'
import hashlib
import json
import os
import sys
from pathlib import Path

claim, contract, kernel, metadata = map(Path, sys.argv[1:])
value = {
    "schema_version": "spec0041.exclusive_launch_claim.v1",
    "authorization": "one_private_label_blind_test_inference_launch",
    "input_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
}
claim.parent.mkdir(parents=True, exist_ok=True)
descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PYSPEC0041CLAIM
    return
  fi

  if [[ "$kernel_dir" == "$mil_training_kernel_dir" \
    || "$kernel_id" == "$mil_training_kernel_id" ]]; then
    if [[ "${KAGGLE_MIL_TRAINING_CONFIRMED:-}" != "1" \
      || "$kernel_id" != "$mil_training_kernel_id" \
      || "$code_file" != "$mil_training_kernel_code_file" ]]; then
      echo "error: exact Spec 0036 MIL training confirmation/path required" >&2
      exit 1
    fi
    local actor expected_input_reference expected_input_contract expected_input_sha256
    actor="$(kaggle_authenticated_username)"
    if [[ "$kernel_dir" == "$mil_training_kernel_dir" ]]; then
      validate_mil_training "$actor"
      expected_input_reference="$actor/$mil_training_dataset_slug"
      expected_input_contract="$mil_training_root/bundle/mil_training_input.json"
      expected_input_sha256=""
    else
      local resume_root resume_receipt
      resume_root="$(dirname "$(dirname "$kernel_dir/run.py")")"
      if [[ "$kernel_dir" != "$resume_root/kernel" ]]; then
        echo "error: Spec 0036 resume push path is not a package kernel" >&2
        exit 1
      fi
      validate_mil_training_resume "$actor" "$resume_root"
      expected_input_reference="$(
        json_field "$resume_root/bundle/mil_training_resume.json" \
          input_dataset_reference
      )"
      expected_input_sha256="$(
        json_field "$resume_root/bundle/mil_training_resume.json" \
          input_contract_sha256
      )"
      expected_input_contract="$mil_training_root/bundle/mil_training_input.json"
      resume_receipt="$resume_root/resume_dataset_receipt.json"
      "$build_python" - \
        "$resume_receipt" \
        "$resume_root/bundle/mil_training_resume.json" <<'PYSPEC0036RESUMEPUSH'
import hashlib
import json
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
contract_path = Path(sys.argv[2])
if not receipt_path.is_file():
    raise SystemExit("Spec 0036 resume push requires its verified dataset receipt")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
bundle = contract_path.parent
files = []
for path in sorted(bundle.rglob("*")):
    if path.is_symlink():
        raise SystemExit("Spec 0036 resume bundle may not contain symlinks")
    if path.is_file() and path.name != "dataset-metadata.json":
        files.append(
            {
                "logical_name": path.relative_to(bundle).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
if (
    receipt.get("schema_version") != "spec0036.resume_dataset_receipt.v1"
    or receipt.get("dataset_reference") != contract.get("dataset_reference")
    or receipt.get("dataset_version") != 1
    or receipt.get("visibility") != "private"
    or receipt.get("status") != "verified"
    or receipt.get("resume_contract_sha256") != contract_sha256
    or receipt.get("remote_files") != files
):
    raise SystemExit("Spec 0036 verified resume receipt binding differs")
PYSPEC0036RESUMEPUSH
    fi
    "$build_python" - \
      "$mil_training_input_receipt" \
      "$expected_input_contract" \
      "$expected_input_reference" \
      "$expected_input_sha256" <<'PYSPEC0036PUSH'
import hashlib
import json
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
contract_path = Path(sys.argv[2])
dataset_reference = sys.argv[3]
expected_sha256 = sys.argv[4]
if not receipt_path.is_file():
    raise SystemExit("Spec 0036 push requires its verified input receipt")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
contract_sha256 = (
    expected_sha256
    if expected_sha256
    else hashlib.sha256(contract_path.read_bytes()).hexdigest()
)
if (
    receipt.get("schema_version") != "spec0036.input_dataset_receipt.v1"
    or receipt.get("dataset_reference") != dataset_reference
    or receipt.get("dataset_version") != 1
    or receipt.get("visibility") != "private"
    or receipt.get("status") != "verified"
    or receipt.get("input_contract_sha256") != contract_sha256
):
    raise SystemExit("Spec 0036 verified input receipt binding differs")
PYSPEC0036PUSH
    if ! grep -q '^SPEC0036_LOCAL_GLOBAL_MIL_TRAINING_READY = True$' \
      "$kernel_dir/$code_file"; then
      echo "error: Spec 0036 MIL training readiness marker differs" >&2
      exit 1
    fi
    return
  fi

  if [[ "$kernel_dir" == "$largest_class_weighted_amp_kernel_dir" \
    || "$kernel_id" == "$largest_class_weighted_amp_kernel_id" ]]; then
    if [[ "${KAGGLE_LARGEST_CLASS_WEIGHTED_AMP_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$largest_class_weighted_amp_kernel_dir" \
      || "$kernel_id" != "$largest_class_weighted_amp_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0035 weighted-AMP probe confirmation/path required" >&2
      exit 1
    fi
    require_build_python
    PYTHONPATH=src "$build_python" scripts/build_largest_class_weighted_amp_probe.py \
      validate --actor maximshtefan >/dev/null
    if ! grep -q '^MAX_OVERFLOW_BACKOFFS = 3$' "$kernel_dir/$code_file"; then
      echo "error: Spec 0035 requires exactly three overflow attempts" >&2
      exit 1
    fi
    return
  fi

  if grep -q 'KAGGLE_FULL_FOREGROUND_COMPLETION_READY = True' "$kernel_dir/$code_file"; then
    if [[ "${KAGGLE_FULL_FOREGROUND_CONFIRMED:-}" != "1" \
      || ! "$kernel_dir" =~ ^runs/local/full_foreground_completion/kernels/run_0[1-8]$ ]]; then
      echo "error: exact full-foreground extraction confirmation/path required" >&2
      exit 1
    fi
    full_foreground_package validate --require-receipt
    return
  fi

  if grep -q 'KAGGLE_WSI45630_CAPACITY_READY = True' "$kernel_dir/$code_file"; then
    if [[ "${KAGGLE_WSI45630_CAPACITY_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "runs/local/wsi45630_capacity/kernel" ]]; then
      echo "error: exact full-WSI capacity confirmation/path required" >&2
      exit 1
    fi
    check_full_wsi_capacity receipt
    return
  fi

  if [[ "$kernel_dir" == "$local_global_capacity_kernel_dir" \
    || "$kernel_id" == "$local_global_capacity_kernel_id" ]]; then
    if [[ "${KAGGLE_LOCAL_GLOBAL_CAPACITY_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$local_global_capacity_kernel_dir" \
      || "$kernel_id" != "$local_global_capacity_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0030 local-global capacity confirmation/path required" >&2
      exit 1
    fi
    if [[ ! -f "$local_global_capacity_initial_claim" \
      || "$(sha256sum "$local_global_capacity_initial_claim" | cut -d' ' -f1)" \
      != "e1b71c22f50dd3d5b171cb3dd8f9d118f902b115a8f52ca89e891175110f32cc" ]]; then
      echo "error: Spec 0030 requires the exact initial failed-attempt claim" >&2
      exit 1
    fi
    if [[ ! -f "$local_global_capacity_rejected_sources_claim" \
      || "$(sha256sum "$local_global_capacity_rejected_sources_claim" | cut -d' ' -f1)" \
      != "8ac6ec81696b3aef5c84407baeb987157310763e75560a2c80be90f1b333e8a6" ]]; then
      echo "error: Spec 0030 requires the exact rejected-sources claim" >&2
      exit 1
    fi
    if [[ -e "$local_global_capacity_claim" ]]; then
      echo "error: Spec 0030 one-run authority was already consumed" >&2
      exit 1
    fi
    if ! grep -q 'spec0030_local_global_capacity_shared_access_retry_authorized' \
      docs/specs/0030-local-global-mil-capacity-probe.md \
      || ! grep -q 'spec0030_local_global_capacity_shared_access_retry_authorized' \
      docs/specs/README.md; then
      echo "error: Spec 0030 remote authorization is not canonical" >&2
      exit 1
    fi
    require_build_python
    "$build_python" scripts/build_wsi45630_local_global_capacity.py \
      validate >/dev/null
    return
  fi

  # This kernel ID and its canonical directory are permanently special. Dispatch
  # on those immutable identities before inspecting mutable source bytes so a
  # removed marker can never fall through to the generic uploader.
  if [[ "$kernel_dir" == "$local_attention_probe_kernel_dir" \
    || "$kernel_id" == "$local_attention_probe_kernel_id" ]]; then
    if [[ "${KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "$local_attention_probe_kernel_dir" \
      || "$kernel_id" != "$local_attention_probe_kernel_id" \
      || "$code_file" != "run.py" ]]; then
      echo "error: exact Spec 0028 local-attention repair confirmation/path required" >&2
      exit 1
    fi
    if [[ -e "$local_attention_repair_probe_push_receipt" ]]; then
      echo "error: Spec 0028 one-run authority was already consumed" >&2
      exit 1
    fi
    if ! grep -q 'spec0028_local_softmax_repair_probe_authorized' \
      docs/specs/0028-wsi45630-local-softmax-repair-probe.md \
      || ! grep -q 'spec0028_local_softmax_repair_probe_authorized' \
      docs/specs/README.md; then
      echo "error: Spec 0028 remote authorization is not canonical" >&2
      exit 1
    fi
    python3 - "$metadata" "$kernel_dir/$code_file" \
      "$local_attention_probe_input_receipt" \
      "$local_attention_probe_push_receipt" \
      "runs/kaggle/wsi45630_local_attention_probe_v1/spec0027_local_attention_probe.json" <<'PYLOCALATTENTIONREPAIR'
import hashlib
import json
import sys
from pathlib import Path

metadata_path, source, input_receipt_path, v1_receipt_path, v1_artifact_path = (
    Path(value) for value in sys.argv[1:]
)
expected_source_sha256 = "8217f9538dd57009f93d0343418a8d3e18d160176d9e3ef0471147ae69c55e12"
expected_metadata_sha256 = "69e3d9abd665a3f438809ef2832fc505431abd46f73f9e00050b6eb40615e8a6"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
assert metadata == {
    "id": "maximusshtefan/eqvae-wsi45630-local-attention-probe",
    "title": "eqvae WSI45630 local attention repair probe",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["maximusshtefan/eqvae-wsi45630-capacity-inputs"],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
code = source.read_text(encoding="utf-8")
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_source_sha256
assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == expected_metadata_sha256
assert source.stat().st_size < 1_000_000
assert "KAGGLE_LOCAL_ATTENTION_REPAIR_PROBE_READY = True" in code
assert "MAX_CORRECTNESS_RELATIVE_L2 = 2e-3" in code
assert "MIN_CORRECTNESS_COSINE = 0.999" in code
compile(code, str(source), "exec")
input_receipt = json.loads(input_receipt_path.read_text(encoding="utf-8"))
assert input_receipt.get("status") == "verified"
assert input_receipt.get("visibility") == "private"
assert input_receipt.get("dataset_version") == 1
assert input_receipt.get("dataset_reference") == metadata["dataset_sources"][0]
assert input_receipt.get("files", {}).get("probe/pointers.csv", {}).get("sha256") == (
    "08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c"
)
v1_receipt = json.loads(v1_receipt_path.read_text(encoding="utf-8"))
assert v1_receipt.get("accepted_version") == 1
assert v1_receipt.get("authority_consumed") is True
assert v1_receipt.get("source_sha256") == (
    "99bf923da39820591b3d5ec388c86fc76a050c931e5b7a0c902982e5dac29f5c"
)
assert hashlib.sha256(v1_receipt_path.read_bytes()).hexdigest() == (
    "a5c531f63e22b09ef4614e167a68dd851d776dd426e010aeb3f18e000aaa6296"
)
assert hashlib.sha256(v1_artifact_path.read_bytes()).hexdigest() == (
    "299dcbfbf03b2301d03cd7fedd109c7833039c5a4a307f00fd56c55d593e7753"
)
PYLOCALATTENTIONREPAIR
    return
  fi

  if grep -q 'KAGGLE_LOCAL_ATTENTION_PROBE_READY = True' "$kernel_dir/$code_file"; then
    if [[ "${KAGGLE_LOCAL_ATTENTION_PROBE_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "kaggle/kernels/wsi45630_local_attention_probe" ]]; then
      echo "error: exact Spec 0027 local-attention probe confirmation/path required" >&2
      exit 1
    fi
    if [[ -e "$local_attention_probe_push_receipt" ]]; then
      echo "error: Spec 0027 one-run authority was already consumed" >&2
      exit 1
    fi
    if ! grep -q 'spec0027_local_softmax_probe_authorized' \
      docs/specs/0027-wsi45630-local-softmax-kernel-probe.md \
      || ! grep -q 'spec0027_local_softmax_probe_authorized' docs/specs/README.md; then
      echo "error: Spec 0027 remote authorization is not canonical" >&2
      exit 1
    fi
    python3 - "$metadata" "$kernel_dir/$code_file" \
      "$local_attention_probe_input_receipt" <<'PYLOCALATTENTION'
import hashlib
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = Path(sys.argv[2])
receipt_path = Path(sys.argv[3])
expected_source_sha256 = "99bf923da39820591b3d5ec388c86fc76a050c931e5b7a0c902982e5dac29f5c"
expected_metadata_sha256 = "9350828a9d59fb7ea0307310c875734e4f8528aa973ed2f6af5f31a4eb1db867"
required = {
    "id": "maximusshtefan/eqvae-wsi45630-local-attention-probe",
    "title": "eqvae WSI45630 local attention probe",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}
for key, expected in required.items():
    assert str(metadata.get(key, "")).lower() == expected.lower(), (key, metadata.get(key))
assert metadata["dataset_sources"] == [
    "maximusshtefan/eqvae-wsi45630-capacity-inputs",
]
for key in ("competition_sources", "kernel_sources", "model_sources"):
    assert metadata[key] == [], key
assert source.stat().st_size < 1_000_000
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_source_sha256
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == (
    expected_metadata_sha256
)
code = source.read_text(encoding="utf-8")
assert "KAGGLE_LOCAL_ATTENTION_PROBE_READY = True" in code
assert "INPUT_CONTRACT_SHA256" in code and "POINTER_SHA256" in code
compile(code, str(source), "exec")
spec = Path("docs/specs/0027-wsi45630-local-softmax-kernel-probe.md").read_text(
    encoding="utf-8",
)
assert expected_source_sha256 in spec
assert expected_metadata_sha256 in spec
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
assert receipt.get("status") == "verified"
assert receipt.get("visibility") == "private"
assert receipt.get("dataset_version") == 1
assert receipt.get("dataset_reference") == metadata["dataset_sources"][0]
assert receipt.get("files", {}).get("probe/pointers.csv", {}).get("sha256") == (
    "08e461846bf16efebac707c82962762f49837916986b29aee0dcd6ca1fc31c6c"
)
PYLOCALATTENTION
    return
  fi

  if grep -q 'KAGGLE_WSI45630_COMPLETION_READY = True' "$kernel_dir/$code_file"; then
    if [[ "${KAGGLE_WSI45630_COMPLETION_CONFIRMED:-}" != "1" \
      || "$kernel_dir" != "runs/local/wsi45630_completion/kernel" ]]; then
      echo "error: exact WSI45630-only extraction confirmation/path required" >&2
      exit 1
    fi
    check_wsi45630_package receipt
    return
  fi

  if [[ "$kernel_dir" == "kaggle/kernels/ubc_ocean_mil_transformer_capacity" ]]; then
    # One-off Spec 0023 synthetic fit probe: never attach latent datasets.
    python3 - "$metadata" "$kernel_dir/$code_file" <<'PYTRANSFORMERCAPACITY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
source = Path(sys.argv[2])
assert metadata["id"] == "maximusshtefan/eqvae-mil-transformer-synthetic-capacity"
assert all(metadata[key] == [] for key in (
    "dataset_sources", "competition_sources", "kernel_sources", "model_sources"
))
assert all(str(metadata[key]).lower() == "true" for key in (
    "is_private", "enable_gpu", "enable_internet"
))
assert metadata["machine_shape"] == "NvidiaTeslaT4"
assert source.stat().st_size < 1_000_000
compile(source.read_text(), str(source), "exec")
PYTRANSFORMERCAPACITY
    return
  fi

  if grep -q "NOT_IMPLEMENTATION_READY" "$kernel_dir/$code_file"; then
    cat >&2 <<'EOF'
error: kernel scaffold is not implementation-ready.

Use docs/behavior_inventory_kaggle.md and spec 0001, implement the real launcher,
and remove the NOT_IMPLEMENTATION_READY guard before pushing.
EOF
    exit 1
  fi

  if grep -q "KAGGLE_SETUP_SMOKE_READY = True" "$kernel_dir/$code_file"; then
    guard_setup_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SYNTHETIC_TIMING_READY = True" "$kernel_dir/$code_file"; then
    guard_synthetic_timing_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True" "$kernel_dir/$code_file"; then
    guard_real_data_runtime_pretest_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_RUNTIME_SELECTION_READY = True" "$kernel_dir/$code_file"; then
    guard_runtime_selection_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True" "$kernel_dir/$code_file"; then
    guard_selected_runtime_debug_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True" "$kernel_dir/$code_file"; then
    guard_selected_runtime_lr_range_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SELECTED_RUNTIME_FULL_READY = True" "$kernel_dir/$code_file"; then
    guard_selected_runtime_full_push_ready "$kernel_dir" "$metadata" "push"
    return
  fi

  if grep -q "KAGGLE_FIXED25_SELECTOR_READY = True" "$kernel_dir/$code_file"; then
    guard_fixed25_selector_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SELECTED_RUNTIME_COMPILE_PROBE_READY = True" "$kernel_dir/$code_file"; then
    guard_selected_runtime_compile_probe_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SO2_ARCHITECTURE_PROBE_READY = True" "$kernel_dir/$code_file"; then
    guard_so2_architecture_probe_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SO2_RUNTIME_READINESS_READY = True" "$kernel_dir/$code_file"; then
    guard_so2_runtime_readiness_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SO2_PRELAUNCH_READY = True" "$kernel_dir/$code_file"; then
    guard_so2_prelaunch_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True" "$kernel_dir/$code_file"; then
    guard_so2_full_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_UBC_OCEAN_TEST_ATLAS_READY = True" "$kernel_dir/$code_file"; then
    guard_ubc_ocean_test_atlas_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "$latent_ready_marker" "$kernel_dir/$code_file"; then
    guard_latent_inference_push_ready "$kernel_dir"
    return
  fi

  if grep -q "$cancer_topup_ready_marker" "$kernel_dir/$code_file"; then
    guard_cancer_topup_push_ready "$kernel_dir"
    return
  fi

  if grep -q "$mil_capacity_probe_ready_marker" "$kernel_dir/$code_file"; then
    guard_mil_capacity_probe_push_ready "$kernel_dir"
    return
  fi

  if grep -q "$supervised_calibration_ready_marker" "$kernel_dir/$code_file"; then
    guard_supervised_calibration_push_ready "$kernel_dir"
    return
  fi

  if [[ ! -f "docs/behavior_inventory_kaggle.md" ]]; then
    echo "error: missing docs/behavior_inventory_kaggle.md" >&2
    exit 1
  fi

  if grep -q "KAGGLE_SMOKE_READY = True" "$kernel_dir/$code_file"; then
    guard_real_smoke_push_ready "$kernel_dir" "$metadata"
    return
  else
    if ! grep -Eq '^Implementation readiness: (locked / implementation-ready|implementation-ready|ready)$' \
      "docs/specs/0001-translatable-normal-vae-baseline.md"; then
      echo "error: spec 0001 is not locked as implementation-ready" >&2
      exit 1
    fi

    if ! grep -Eq '^\| `0001-translatable-normal-vae-baseline\.md` \|[^|]*locked / implementation-ready' \
      "docs/specs/README.md"; then
      echo "error: spec 0001 is not locked as implementation-ready in docs/specs/README.md" >&2
      exit 1
    fi
  fi
}

record_local_attention_probe_push() {
  local kernel_dir="$1"
  local accepted_version="$2"
  local receipt="$local_attention_probe_push_receipt"
  mkdir -p "$(dirname "$receipt")"
  python3 - "$receipt" "$kernel_dir/kernel-metadata.json" "$kernel_dir/run.py" \
    "$accepted_version" <<'PYLOCALATTENTIONRECEIPT'
import datetime
import hashlib
import json
import sys
from pathlib import Path

receipt = Path(sys.argv[1])
metadata = Path(sys.argv[2])
source = Path(sys.argv[3])
payload = {
    "schema_version": "spec0027.push_receipt.v1",
    "authority_consumed": True,
    "kernel_id": "maximusshtefan/eqvae-wsi45630-local-attention-probe",
    "accepted_version": int(sys.argv[4]),
    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "recorded_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
with receipt.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PYLOCALATTENTIONRECEIPT
}

make_local_attention_repair_probe_snapshot() {
  local kernel_dir="$1"
  local snapshot
  snapshot="$(mktemp -d "$TMPDIR/spec0028_upload.XXXXXX")" || return 1
  cp "$kernel_dir/run.py" "$snapshot/run.py" || return 1
  cp "$kernel_dir/kernel-metadata.json" "$snapshot/kernel-metadata.json" \
    || return 1
  if ! python3 - "$snapshot/run.py" "$snapshot/kernel-metadata.json" <<'PYLOCALATTENTIONREPAIRSNAPSHOT'
import hashlib
import sys
from pathlib import Path

source, metadata = (Path(value) for value in sys.argv[1:])
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "8217f9538dd57009f93d0343418a8d3e18d160176d9e3ef0471147ae69c55e12"
)
assert hashlib.sha256(metadata.read_bytes()).hexdigest() == (
    "69e3d9abd665a3f438809ef2832fc505431abd46f73f9e00050b6eb40615e8a6"
)
PYLOCALATTENTIONREPAIRSNAPSHOT
  then
    return 1
  fi
  printf '%s\n' "$snapshot"
}

claim_local_attention_repair_probe_push() {
  local kernel_dir="$1"
  local receipt="$local_attention_repair_probe_push_receipt"
  mkdir -p "$(dirname "$receipt")"
  python3 - "$receipt" "$kernel_dir/kernel-metadata.json" "$kernel_dir/run.py" \
    <<'PYLOCALATTENTIONREPAIRCLAIM'
import datetime
import hashlib
import json
import sys
from pathlib import Path

receipt = Path(sys.argv[1])
metadata = Path(sys.argv[2])
source = Path(sys.argv[3])
source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
metadata_sha256 = hashlib.sha256(metadata.read_bytes()).hexdigest()
assert source_sha256 == (
    "8217f9538dd57009f93d0343418a8d3e18d160176d9e3ef0471147ae69c55e12"
)
assert metadata_sha256 == (
    "69e3d9abd665a3f438809ef2832fc505431abd46f73f9e00050b6eb40615e8a6"
)
payload = {
    "schema_version": "spec0028.push_attempt.v1",
    "authority_consumed": True,
    "kernel_id": "maximusshtefan/eqvae-wsi45630-local-attention-probe",
    "status": "attempt_claimed",
    "accepted_version": None,
    "metadata_sha256": metadata_sha256,
    "source_sha256": source_sha256,
    "attempt_started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
with receipt.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PYLOCALATTENTIONREPAIRCLAIM
}

finalize_local_attention_repair_probe_push() {
  local kernel_dir="$1"
  local accepted_version="$2"
  local receipt="$local_attention_repair_probe_push_receipt"
  python3 - "$receipt" "$kernel_dir/kernel-metadata.json" "$kernel_dir/run.py" \
    "$accepted_version" <<'PYLOCALATTENTIONREPAIRRECEIPT'
import datetime
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

receipt = Path(sys.argv[1])
metadata = Path(sys.argv[2])
source = Path(sys.argv[3])
accepted_version = int(sys.argv[4])
claim = json.loads(receipt.read_text(encoding="utf-8"))
source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
metadata_sha256 = hashlib.sha256(metadata.read_bytes()).hexdigest()
assert claim.get("schema_version") == "spec0028.push_attempt.v1"
assert claim.get("authority_consumed") is True
assert claim.get("status") == "attempt_claimed"
assert claim.get("accepted_version") is None
assert claim.get("source_sha256") == source_sha256
assert claim.get("metadata_sha256") == metadata_sha256
payload = {
    "schema_version": (
        "spec0028.push_receipt.v1"
        if accepted_version == 2
        else "spec0028.push_attempt.v1"
    ),
    "authority_consumed": True,
    "kernel_id": "maximusshtefan/eqvae-wsi45630-local-attention-probe",
    "status": "accepted" if accepted_version == 2 else "unexpected_version",
    "accepted_version": accepted_version,
    "metadata_sha256": metadata_sha256,
    "source_sha256": source_sha256,
    "attempt_started_utc": claim["attempt_started_utc"],
    "recorded_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
with tempfile.NamedTemporaryFile(
    "w",
    encoding="utf-8",
    dir=receipt.parent,
    prefix=f".{receipt.name}.",
    delete=False,
) as handle:
    temporary = Path(handle.name)
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, receipt)
if accepted_version != 2:
    raise SystemExit("Spec 0028 expected Kaggle kernel version 2")
PYLOCALATTENTIONREPAIRRECEIPT
}

guard_ubc_ocean_test_atlas_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if ! cmp -s "$ubc_ocean_test_generator" "$kernel_dir/run.py"; then
    echo "error: atlas run.py is stale; rebuild it before push" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PYATLASMETA'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
errors: list[str] = []
required = {
    "id": "maximusshtefan/eqvae-ubc-ocean-test-atlas",
    "title": "eqvae UBC-OCEAN test atlas",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_internet": "true",
}
for key, expected in required.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")
if data.get("competition_sources") != ["UBC-OCEAN"]:
    errors.append("competition_sources must contain only UBC-OCEAN")
mask_source = "sohier/ubc-ovarian-cancer-competition-supplemental-masks"
checkpoint_source = "maximusshtefan/eqvae-ubc-ocean-test-atlas-checkpoint"
allowed_dataset_sources = ([mask_source], [mask_source, checkpoint_source])
if data.get("dataset_sources") not in allowed_dataset_sources:
    errors.append(
        "dataset_sources must contain the official masks and, only for resume, "
        "the exact private atlas-checkpoint dataset"
    )
for source_field in ("kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be empty")
if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PYATLASMETA
}

guard_real_smoke_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if ! grep -q 'kaggle_smoke_ready' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 does not authorize the narrow Kaggle smoke" >&2
    exit 1
  fi
  if ! grep -Eq '^\| `0001-translatable-normal-vae-baseline\.md` \|[^|]*kaggle smoke is `kaggle_smoke_ready`' \
    "docs/specs/README.md"; then
    echo "error: spec index does not authorize the narrow Kaggle smoke" >&2
    exit 1
  fi

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SMOKE_READY = True" \
    --verify-only

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/non-eq-vae-debug",
    "title": "non-eq-VAE debug",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

dataset_sources = data.get("dataset_sources")
expected_dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
forbidden_sources = {"maximusshtefan/non-eq-vae-output"}

if dataset_sources != expected_dataset_sources:
    errors.append(
        "dataset_sources must be exactly "
        f"{expected_dataset_sources!r} for the spec 0001 debug kernel"
    )

for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for the spec 0001 debug kernel")

for source_group in (
    data.get("dataset_sources"),
    data.get("competition_sources"),
    data.get("kernel_sources"),
    data.get("model_sources"),
):
    if isinstance(source_group, list):
        for source in source_group:
            if source in forbidden_sources:
                errors.append(f"forbidden historical FSQ source: {source!r}")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  python3 - "$kernel_dir/run.py" <<'PY'
import base64
import io
import json
import re
import sys
import zipfile

run_text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: generated run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)
zip_bytes = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
    config = json.loads(
        archive.read("configs/spec0001/non_eq_vae_kaggle_debug.json"),
    )

smoke = config.get("kaggle_smoke")
errors: list[str] = []
if not isinstance(smoke, dict):
    errors.append("payload config must contain kaggle_smoke object")
else:
    expected = {
        "full_run_eligible": False,
        "batch_size": 1,
        "max_validation_batches": 1,
        "num_workers": 0,
    }
    for key, value in expected.items():
        if smoke.get(key) != value:
            errors.append(f"kaggle_smoke.{key} must be {value!r}")
    max_train_steps = smoke.get("max_train_steps")
    if not isinstance(max_train_steps, int) or not 1 <= max_train_steps <= 3:
        errors.append("kaggle_smoke.max_train_steps must be an integer from 1 to 3")
    if smoke.get("benchmark_source") != "kaggle_script_kernel_capped_smoke":
        errors.append("kaggle_smoke.benchmark_source must identify capped smoke")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  for required_hook in single_visible_t4 dual_t4_ddp wrong_accelerator; do
    if ! grep -q "$required_hook" "$kernel_dir/run.py"; then
      echo "error: launcher must include $required_hook runtime validation hook" >&2
      exit 1
    fi
  done
}

guard_setup_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: setup smoke must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if ! grep -q 'kaggle_setup_smoke_ready' \
    "docs/specs/0003-kaggle-cli-execution-workflow.md"; then
    echo "error: spec 0003 does not authorize the synthetic setup smoke" >&2
    exit 1
  fi

  if ! grep -q 'synthetic no-dataset setup smoke' \
    "docs/kaggle_cli_workflow.md"; then
    echo "error: Kaggle workflow doc does not describe setup-smoke evidence" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/eqvae-setup-smoke",
    "title": "eqvae setup smoke",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_internet": "true",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

if data.get("machine_shape") not in (None, "", "None"):
    errors.append("setup smoke machine_shape must be absent or empty")

for source_field in (
    "dataset_sources",
    "competition_sources",
    "kernel_sources",
    "model_sources",
):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for setup smoke")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  if ! grep -q "synthetic_kaggle_setup_smoke" "$kernel_dir/run.py"; then
    echo "error: setup run.py must declare synthetic_kaggle_setup_smoke" >&2
    exit 1
  fi

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SETUP_SMOKE_READY = True" \
    --verify-only
}

guard_fixed25_selector_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: fixed25 selector must be a single generated run.py, not a payload" >&2
    exit 1
  fi

  if ! grep -q 'fixed25_selector_kernel_ready' \
    "docs/specs/0010-fixed25-equivariance-artifact-protocol.md"; then
    echo "error: spec 0010 does not authorize the fixed25 selector kernel" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_FULL_DATASET_CONFIRMED=1 to attach the UBC dataset for selector generation" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/eqvae-fixed25-selector",
    "title": "eqvae fixed25 selector",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_internet": "true",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

if data.get("machine_shape") not in (None, "", "None"):
    errors.append("fixed25 selector machine_shape must be absent or empty (CPU-only)")

expected_datasets = [
    "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    "maximusshtefan/eqvae-baseline-session1-step15000",
]
if data.get("dataset_sources") != expected_datasets:
    errors.append("dataset_sources must attach the exact UBC and session-1 datasets")

for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  local code_file
  code_file="$(json_field "$metadata" code_file)"
  local required_text
  for required_text in \
    "KAGGLE_FIXED25_SELECTOR_READY = True" \
    "fixed_25_validation" \
    "select_fixed_patches" \
    "fixed25_originals" \
    "originals.pt" \
    "originals.png" \
    "--validate-crc"; do
    if ! grep -q -- "$required_text" "$kernel_dir/$code_file"; then
      echo "error: fixed25 selector run.py missing required text: $required_text" >&2
      exit 1
    fi
  done

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_FIXED25_SELECTOR_READY = True" \
    --verify-only
}

guard_synthetic_timing_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: synthetic timing must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" == "1" ]]; then
    echo "error: do not set KAGGLE_FULL_DATASET_CONFIRMED=1 for no-dataset synthetic timing" >&2
    exit 1
  fi

  if ! grep -q 'kaggle_synthetic_timing_contract_ready' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 does not authorize the synthetic timing contract" >&2
    exit 1
  fi
  if ! grep -q 'synthetic binary timing pretest contract is `kaggle_synthetic_timing_contract_ready`' \
    "docs/specs/README.md"; then
    echo "error: spec index does not authorize the synthetic timing contract" >&2
    exit 1
  fi
  if ! grep -q 'The synthetic binary timing pretest workflow becomes Kaggle-push-ready' \
    "docs/specs/0003-kaggle-cli-execution-workflow.md"; then
    echo "error: spec 0003 does not describe synthetic timing push readiness" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/eqvae-synthetic-timing",
    "title": "eqvae synthetic timing",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

for source_field in (
    "dataset_sources",
    "competition_sources",
    "kernel_sources",
    "model_sources",
):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for synthetic timing")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SYNTHETIC_TIMING_READY = True" \
    --verify-only

  local run_file="$kernel_dir/run.py"
  python3 - "$run_file" <<'PY'
import base64
import io
import re
import sys
import zipfile
from pathlib import Path

run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: synthetic timing run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    try:
        source = archive.read(
            "src/eqvae/benchmarking/synthetic_timing.py",
        ).decode("utf-8")
    except KeyError:
        print(
            "error: synthetic timing payload is missing synthetic_timing.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

required_source_text = (
    'DEFAULT_PROFILE_NAME = "synthetic_binary_2gib_histology_like_v1"',
    "DEFAULT_TOTAL_PATCHES = 10_912",
    "DEFAULT_SPLIT_PATCHES = 5_456",
    'COMPACT_PROFILE_NAME = "synthetic_binary_0p81gb_histology_like_v1"',
    "COMPACT_TOTAL_PATCHES = 4_096",
    "COMPACT_SPLIT_PATCHES = 2_048",
    "def compact_synthetic_timing_profile()",
    "REPEAT_SHORTLIST_WARMUP_STEPS = 5",
    "REPEAT_SHORTLIST_MEASURED_STEPS = 25",
    "def repeat_shortlist_row_specs()",
)
missing = [text for text in required_source_text if text not in source]
if missing:
    for text in missing:
        print(
            f"error: synthetic timing embedded source missing required text: {text}",
            file=sys.stderr,
        )
    raise SystemExit(1)
PY

  if grep -q "selected_runtime" "$run_file"; then
    echo "error: synthetic timing launcher must not reference selected runtime artifacts" >&2
    exit 1
  fi

  for required_text in \
    "synthetic_timing_manifest.json" \
    "synthetic_timing_runtime_proof.json" \
    "synthetic_timing_matrix.csv" \
    "synthetic_timing_recommendations.json" \
    "non_promotable_synthetic_timing" \
    "kaggle_synthetic_timing_pretest" \
    "kaggle_no_dataset_generated_ubc_shards" \
    "blocked_claims" \
    "/kaggle/working" \
    "single_visible_t4" \
    "dual_t4_ddp" \
    "eqvae_synthetic_timing_repeat_shortlist" \
    "repeat_shortlist_row_specs" \
    "SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST" \
    "wrong_accelerator"; do
    if ! grep -q "$required_text" "$run_file"; then
      echo "error: synthetic timing run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

guard_selected_runtime_compile_probe_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: compile probe must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" == "1" ]]; then
    echo "error: do not set KAGGLE_FULL_DATASET_CONFIRMED=1 for the no-dataset compile probe" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/eqvae-selected-runtime-compile-probe",
    "title": "eqvae selected runtime compile probe",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

for source_field in (
    "dataset_sources",
    "competition_sources",
    "kernel_sources",
    "model_sources",
):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for the compile probe")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SELECTED_RUNTIME_COMPILE_PROBE_READY = True" \
    --verify-only

  local run_file="$kernel_dir/run.py"
  python3 - "$run_file" <<'PY'
import base64
import io
import re
import sys
import zipfile
from pathlib import Path

run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: compile probe run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    try:
        source = archive.read(
            "src/eqvae/benchmarking/compiled_fastpath_probe.py",
        ).decode("utf-8")
    except KeyError:
        print(
            "error: compile probe payload is missing compiled_fastpath_probe.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

required_source_text = (
    'COMPILED_FASTPATH_PROBE_KIND = "kaggle_compiled_fastpath_probe"',
    'COMPILED_FASTPATH_PROBE_STATUS_SCOPE = "non_promotable_compiled_fastpath_probe"',
    'RECIPE_PYTHON_REDUCER = "python_reducer_whole_step"',
    'RECIPE_DDP_OPTIMIZER = "ddp_optimizer_whole_step"',
    "def run_compiled_fastpath_probe(",
    "def run_negative_control_desync(",
)
missing = [text for text in required_source_text if text not in source]
if missing:
    for text in missing:
        print(
            f"error: compile probe embedded source missing required text: {text}",
            file=sys.stderr,
        )
    raise SystemExit(1)
PY

  for required_text in \
    "compiled_fastpath_probe_proof.json" \
    "compiled_fastpath_probe_matrix.csv" \
    "compiled_fastpath_probe_manifest.json" \
    "non_promotable_compiled_fastpath_probe" \
    "kaggle_compiled_fastpath_probe" \
    "kaggle_no_dataset_synthetic_compiled_fastpath" \
    "blocked_claims" \
    "/kaggle/working" \
    "torch.distributed.run" \
    "--nproc_per_node=2" \
    "eqvae.benchmarking.compiled_fastpath_probe"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: compile probe run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

guard_so2_architecture_probe_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  local mode="${3:-push}"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: SO(2) probe must be one generated run.py, not a sibling payload" >&2
    exit 1
  fi
  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" == "1" ]]; then
    echo "error: do not attach a dataset to the generated-tensor SO(2) probe" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PYSO2METADATA'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "id": "maximusshtefan/eqvae-so2-architecture-probe",
    "title": "eqvae so2 architecture probe",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}
errors = [
    f"{key} must be {value!r}"
    for key, value in expected.items()
    if (
        str(data.get(key, "")).lower()
        if value in {"true", "false"}
        else str(data.get(key, ""))
    ) != value
]
for field in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
    if data.get(field) != []:
        errors.append(f"{field} must be empty")
if errors:
    raise SystemExit("\n".join(f"error: {error}" for error in errors))
PYSO2METADATA

  local verify_args=(
    --kernel-dir "$kernel_dir"
    --ready-marker "KAGGLE_SO2_ARCHITECTURE_PROBE_READY = True"
    --verify-only
  )
  if [[ "$mode" == "local_preflight" ]]; then
    verify_args+=(--allow-dirty)
  elif [[ "$mode" != "push" ]]; then
    echo "error: unsupported SO(2) probe guard mode: $mode" >&2
    exit 1
  fi
  build_kernel_py "${verify_args[@]}"

  local run_file="$kernel_dir/run.py"
  for required_text in \
    "spec0013_so2_dual_t4_probe.json" \
    "spec0013.so2_dual_t4_final.v1" \
    "locked_so2_architecture_mechanics_final" \
    "padded_bmm_direct" \
    "compile_step_python_reducer_fp16_channels_last" \
    "e9e998fd161f0955959c64aed7cd7ddbdfcb55a271b9ce05805903c97c93efb8" \
    "torch.distributed.run" \
    "--nproc_per_node=2" \
    "eqvae.benchmarking.so2_architecture_probe" \
    "graph_breaks,recompiles"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: SO(2) probe run.py missing required text: $required_text" >&2
      exit 1
    fi
  done

  python3 - "$run_file" <<'PYSO2PAYLOAD'
import base64
import io
import re
import sys
import zipfile
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""', text, re.DOTALL)
if match is None:
    raise SystemExit("error: SO(2) probe has no embedded payload")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group("payload")))) as archive:
    source = archive.read("src/eqvae/benchmarking/so2_architecture_probe.py").decode()
required = (
    "PER_DEVICE_BATCH: Final = 4",
    "SETTLED_UPDATES: Final = 32",
    "WARMUP_UPDATES: Final = 20",
    "TIMED_WINDOW_UPDATES: Final = 50",
    'RUNTIME_BUNDLE_ID: Final = "compile_step_python_reducer_fp16_channels_last"',
    "SO2LargestDDConv",
    "def _gradient_mean_check(",
    "def _check_buffers_across_ranks(",
)
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("\n".join(f"error: embedded SO(2) source missing {item}" for item in missing))
for forbidden in ("four_mm_three_cat", "four_mm_direct", "_follow_up_verdict"):
    if forbidden in source:
        raise SystemExit(f"error: embedded final SO(2) source retains {forbidden}")
PYSO2PAYLOAD
}

preflight_so2_architecture_probe() {
  build_embedded_kernel "$so2_architecture_probe_kernel_dir"
  guard_so2_architecture_probe_push_ready \
    "$so2_architecture_probe_kernel_dir" \
    "$(metadata_path "$so2_architecture_probe_kernel_dir")" \
    "local_preflight"
  echo "ok: Spec 0013 dual-T4 probe is built and locally guarded; no remote write performed"
}

guard_so2_runtime_readiness_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  local mode="${3:-push}"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: SO(2) readiness must be one generated run.py" >&2
    exit 1
  fi
  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" == "1" ]]; then
    echo "error: do not attach a dataset to SO(2) readiness" >&2
    exit 1
  fi
  python3 - "$metadata" <<'PYSO2READINESSMETADATA'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "id": "maximusshtefan/eqvae-so2-runtime-readiness",
    "title": "eqvae so2 runtime readiness",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}
errors = [
    f"{key} must be {value!r}"
    for key, value in expected.items()
    if (str(data.get(key, "")).lower() if value in {"true", "false"} else str(data.get(key, ""))) != value
]
for field in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
    if data.get(field) != []:
        errors.append(f"{field} must be empty")
if errors:
    raise SystemExit("\n".join(f"error: {error}" for error in errors))
PYSO2READINESSMETADATA

  local verify_args=(
    --kernel-dir "$kernel_dir"
    --ready-marker "KAGGLE_SO2_RUNTIME_READINESS_READY = True"
    --verify-only
  )
  if [[ "$mode" == "local_preflight" ]]; then
    verify_args+=(--allow-dirty)
  elif [[ "$mode" != "push" ]]; then
    echo "error: unsupported SO(2) readiness guard mode: $mode" >&2
    exit 1
  fi
  build_kernel_py "${verify_args[@]}"

  local run_file="$kernel_dir/run.py"
  for required_text in \
    "spec0015_so2_runtime_readiness.json" \
    "spec0015_so2_gate_health.csv" \
    "spec0015.so2_selected_runtime_readiness.v1" \
    "compile_step_python_reducer_fp16_channels_last" \
    "generated_device_resident" \
    "GATE_ROW_COUNT = 68" \
    "torch.distributed.run" \
    "--nproc_per_node=2" \
    "eqvae.benchmarking.so2_runtime_readiness" \
    "graph_breaks,recompiles"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: SO(2) readiness run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

preflight_so2_runtime_readiness() {
  build_embedded_kernel "$so2_runtime_readiness_kernel_dir"
  guard_so2_runtime_readiness_push_ready \
    "$so2_runtime_readiness_kernel_dir" \
    "$(metadata_path "$so2_runtime_readiness_kernel_dir")" \
    "local_preflight"
  echo "ok: Spec 0015 SO(2) readiness is built and guarded; no remote write performed"
}

guard_so2_training_metadata() {
  local metadata="$1"
  local expected_id="$2"
  local resume_dataset_slug="${3:-}"
  python3 - "$metadata" "$expected_id" "$resume_dataset_slug" <<'PYSO2TRAINMETADATA'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_id = sys.argv[2]
resume_dataset_slug = sys.argv[3]
dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
if resume_dataset_slug:
    dataset_sources.append(resume_dataset_slug)
expected = {
    "id": expected_id,
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": dataset_sources,
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
errors = [f"{key} must be {value!r}" for key, value in expected.items() if data.get(key) != value]
if errors:
    raise SystemExit("\n".join(f"error: {error}" for error in errors))
PYSO2TRAINMETADATA
}

guard_so2_prelaunch_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  guard_so2_training_metadata "$metadata" "maximusshtefan/eqvae-so2-prelaunch"
  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SO2_PRELAUNCH_READY = True" \
    --verify-only
}

guard_so2_full_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  local mode="${3:-push}"
  guard_so2_training_metadata \
    "$metadata" \
    "maximshtefan/eqvae-so2-selected-runtime-full-session7" \
    "$so2_full_resume_dataset_slug"
  local verify_args=(
    --kernel-dir "$kernel_dir"
    --ready-marker "KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True"
    --verify-only
  )
  if [[ "$mode" == "local_preflight" ]]; then
    verify_args+=(--allow-dirty)
  elif [[ "$mode" != "push" ]]; then
    echo "error: unsupported SO2 full guard mode: $mode" >&2
    exit 1
  fi
  build_kernel_py "${verify_args[@]}"
  if [[ "${KAGGLE_SO2_FULL_COST_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_SO2_FULL_COST_CONFIRMED=1 after accepting measured prelaunch cost" >&2
    exit 1
  fi
  local verdict="runs/kaggle/so2_prelaunch/benchmark/so2_prelaunch_verdict.json"
  if [[ ! -f "$verdict" ]]; then
    echo "error: missing downloaded SO2 prelaunch verdict: $verdict" >&2
    exit 1
  fi
  PYTHONPATH=src .venv/bin/python - \
    "$verdict" \
    "$so2_full_session1_output_dir/embedded_payload" \
    "$so2_full_resume_authority_dir/embedded_payload" \
    "$so2_full_resume_dataset_dir" <<'PYSO2FULLVERDICT'
import hashlib
import json
import sys
from pathlib import Path
from eqvae.benchmarking.so2_prelaunch import (
    execution_identity,
    validate_prelaunch_artifacts,
)
from eqvae.checkpointing import read_training_checkpoint_metadata
from eqvae.config import resolve_json_config

EXPECTED_PRELAUNCH_COMMIT = "4aaf614f2cdbf1bc628e13858eb6c4e08300266b"
EXPECTED_RESUME_COMMIT = "396d897dc442b5e5f9f94e32f01679d35fa69858"
EXPECTED_DATASET_SLUG = "maximshtefan/eqvae-so2-session6-step54000"
EXPECTED_CHECKPOINT_SHA256 = (
    "2ae4785571e2d1b4e690957e3cf74f749c7e273f1701ee274cc7b2b2e4a8742c"
)
EXPECTED_CHECKPOINT_BYTES = 16_440_368
EXPECTED_CONTINUATION_WRAPPER_SHA256 = (
    "03887128886879b8c2ac4e68b210233dcbcbc181344c224a3990bb34824a8dd0"
)
EXPECTED_STEP = 54000
ALLOWED_CONTINUATION_CHANGES = {
    "kaggle/kernels/so2_selected_runtime_full/kernel-metadata.json",
    "kaggle/kernels/so2_selected_runtime_full/run_template.py",
}

verdict = Path(sys.argv[1])
prelaunch_authority = Path(sys.argv[2])
resume_authority = Path(sys.argv[3])
dataset_dir = Path(sys.argv[4])
repo = Path.cwd()

blockers = list(
    validate_prelaunch_artifacts(
        verdict.parents[1],
        repo_root=prelaunch_authority,
        expected_source_commit=EXPECTED_PRELAUNCH_COMMIT,
    ),
)
resume_manifest = json.loads(
    (resume_authority / "payload_manifest.json").read_text(encoding="utf-8")
)
if (
    resume_manifest.get("git_commit") != EXPECTED_RESUME_COMMIT
    or resume_manifest.get("git_dirty") is not False
):
    blockers.append("so2_continuation_resume_authority_commit_mismatch")
prelaunch_identity = execution_identity(prelaunch_authority)
resume_identity = execution_identity(resume_authority)
current_identity = execution_identity(repo)
continuation_wrapper = (
    repo / "kaggle/kernels/so2_selected_runtime_full/run_template.py"
)
if hashlib.sha256(continuation_wrapper.read_bytes()).hexdigest() != (
    EXPECTED_CONTINUATION_WRAPPER_SHA256
):
    blockers.append("so2_continuation_wrapper_sha256_mismatch")
for name, expected in prelaunch_identity.items():
    if name in ALLOWED_CONTINUATION_CHANGES:
        continue
    if resume_identity.get(name) != expected:
        blockers.append(f"so2_continuation_resume_execution_core_changed:{name}")
    if current_identity.get(name) != expected:
        blockers.append(f"so2_continuation_execution_core_changed:{name}")

expected_files = {"dataset-metadata.json", "step_054000.pt"}
observed_files = (
    {path.name for path in dataset_dir.iterdir()}
    if dataset_dir.is_dir()
    else set()
)
if observed_files != expected_files:
    blockers.append("so2_continuation_dataset_files_mismatch")
checkpoint = dataset_dir / "step_054000.pt"
if checkpoint.is_file():
    observed_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if observed_sha256 != EXPECTED_CHECKPOINT_SHA256:
        blockers.append("so2_continuation_checkpoint_sha256_mismatch")
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        blockers.append("so2_continuation_checkpoint_size_mismatch")
    metadata = read_training_checkpoint_metadata(path=checkpoint)
    if (
        metadata.optimizer_step != EXPECTED_STEP
        or metadata.successful_optimizer_update_count != EXPECTED_STEP
    ):
        blockers.append("so2_continuation_checkpoint_step_mismatch")
    runtime = repo / "configs/spec0001/non_eq_vae_selected_runtime.json"
    runtime_sha256 = hashlib.sha256(runtime.read_bytes()).hexdigest()
    effective_sha256 = resolve_json_config(
        repo / "configs/spec0016/so2_selected_runtime_full.json",
    ).effective_config_hash
    if metadata.runtime_config_sha256 != runtime_sha256:
        blockers.append("so2_continuation_checkpoint_runtime_mismatch")
    if metadata.effective_config_sha256 != effective_sha256:
        blockers.append("so2_continuation_checkpoint_config_mismatch")
else:
    blockers.append("so2_continuation_checkpoint_missing")

dataset_metadata = dataset_dir / "dataset-metadata.json"
if dataset_metadata.is_file():
    payload = json.loads(dataset_metadata.read_text(encoding="utf-8"))
    if (
        payload.get("id") != EXPECTED_DATASET_SLUG
        or EXPECTED_CHECKPOINT_SHA256 not in payload.get("description", "")
        or str(EXPECTED_CHECKPOINT_BYTES) not in payload.get("description", "")
    ):
        blockers.append("so2_continuation_dataset_metadata_mismatch")
else:
    blockers.append("so2_continuation_dataset_metadata_missing")

if blockers:
    raise SystemExit("\n".join(f"error: {blocker}" for blocker in blockers))
PYSO2FULLVERDICT
}

preflight_so2_prelaunch() {
  build_embedded_kernel "$so2_prelaunch_kernel_dir"
  guard_so2_training_metadata \
    "$(metadata_path "$so2_prelaunch_kernel_dir")" \
    "maximusshtefan/eqvae-so2-prelaunch"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q \
    tests/test_so2_prelaunch.py tests/test_so2_full_run.py
}

preflight_so2_full() {
  build_embedded_kernel "$so2_full_kernel_dir"
  KAGGLE_SO2_FULL_COST_CONFIRMED=1 guard_so2_full_push_ready \
    "$so2_full_kernel_dir" \
    "$(metadata_path "$so2_full_kernel_dir")" \
    "local_preflight"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q \
    tests/test_so2_prelaunch.py tests/test_so2_full_run.py
}

guard_real_data_runtime_pretest_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: real-data runtime pretest must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOF'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the real-data runtime pretest.
EOF
    exit 1
  fi

  if ! grep -q 'real_data_runtime_pretest_contract_ready' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 does not authorize the real-data runtime pretest contract" >&2
    exit 1
  fi
  if ! grep -q 'real_data_runtime_pretest_contract_ready' \
    "docs/specs/README.md"; then
    echo "error: spec index does not authorize the real-data runtime pretest contract" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/eqvae-real-data-runtime-pretest",
    "title": "eqvae real data runtime pretest",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

expected_dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
if data.get("dataset_sources") != expected_dataset_sources:
    errors.append(f"dataset_sources must be exactly {expected_dataset_sources!r}")

for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for real-data pretest")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True" \
    --verify-only

  local run_file="$kernel_dir/run.py"
  python3 - "$run_file" <<'PY'
import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: real-data pretest run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    try:
        source = archive.read(
            "src/eqvae/benchmarking/real_data_runtime_pretest.py",
        ).decode("utf-8")
        config = json.loads(
            archive.read(
                "configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json",
            ),
        )
    except KeyError as error:
        print(f"error: real-data pretest payload missing {error}", file=sys.stderr)
        raise SystemExit(1) from None

errors: list[str] = []
if "write_synthetic_benchmark_artifacts" in source:
    errors.append("real-data pretest payload must not call schema selected-runtime writer")
if "selected_runtime_path" in source:
    errors.append("real-data pretest payload must not define selected_runtime_path")
if "def _reject_selected_runtime_artifact" not in source:
    errors.append("real-data pretest payload must reject stale selected_runtime artifacts")
if source.count("_reject_selected_runtime_artifact(") < 3:
    errors.append("real-data pretest payload must check selected_runtime before and after writes")
if re.search(r"write_json\s*\([^)]*selected_runtime", source, flags=re.DOTALL):
    errors.append("real-data pretest payload must not write selected_runtime artifacts")

data = config.get("data")
runtime = config.get("runtime_matrix")
pretest = config.get("runtime_pretest")
if config.get("status") != "real_data_runtime_pretest_kernel_guard_ready_non_promotable":
    errors.append(
        "config.status must be real_data_runtime_pretest_kernel_guard_ready_non_promotable",
    )
if not isinstance(data, dict):
    errors.append("config.data must be an object")
else:
    if data.get("dataset_slug") != "maximusshtefan/patches-pre-shuffled-ubc-ocean":
        errors.append("config.data.dataset_slug must be the pre-shuffled patch dataset")
    cap = data.get("benchmark_cap")
    if not isinstance(cap, dict):
        errors.append("config.data.benchmark_cap must be an object")
    else:
        if cap.get("train_patch_count") != 8192:
            errors.append("benchmark_cap.train_patch_count must be 8192")
        if cap.get("validation_patch_count") != 2048:
            errors.append("benchmark_cap.validation_patch_count must be 2048")
        if cap.get("full_epoch_allowed") is not False:
            errors.append("benchmark_cap.full_epoch_allowed must be false")
if not isinstance(runtime, dict):
    errors.append("config.runtime_matrix must be an object")
else:
    settle = runtime.get("compile_settle_policy")
    if not isinstance(settle, dict) or settle.get("compile_settle_steps") != 5:
        errors.append("compile_settle_steps must be 5")
if not isinstance(pretest, dict):
    errors.append("config.runtime_pretest must be an object")
else:
    if pretest.get("full_run_eligible") is not False:
        errors.append("runtime_pretest.full_run_eligible must be false")
    if pretest.get("writes_selected_runtime") is not False:
        errors.append("runtime_pretest.writes_selected_runtime must be false")
    artifacts = pretest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("runtime_pretest.artifacts must be an object")
    elif "selected_runtime" in artifacts:
        errors.append("runtime_pretest.artifacts must not include selected_runtime")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  for required_text in \
    "real_data_runtime_pretest_manifest.json" \
    "runtime_proof.json" \
    "runtime_matrix.csv" \
    "dataloader_matrix.csv" \
    "numerical_checks.csv" \
    "corruption_checks.csv" \
    "gate_health_summary.json" \
    "real_data_runtime_pretest_recommendations.json" \
    "phase_timings.json" \
    "non_promotable_real_data_runtime_pretest" \
    "real_data_runtime_pretest" \
    "blocked_claims" \
    "selected_runtime.json" \
    "single_visible_t4" \
    "dual_t4_ddp" \
    "wrong_accelerator"; do
    if ! grep -q "$required_text" "$run_file"; then
      echo "error: real-data runtime pretest run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

guard_runtime_selection_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: runtime selection must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOF'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the selected-runtime benchmark.
EOF
    exit 1
  fi

  if ! grep -q 'v8_shortlist_eager_amp_then_dual_gate' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 does not describe the v8 selected-runtime slice" >&2
    exit 1
  fi
  if ! grep -q 'runtime_selection_kernel_ready' \
    "docs/specs/0003-kaggle-cli-execution-workflow.md"; then
    echo "error: spec 0003 does not authorize runtime-selection kernel push readiness" >&2
    exit 1
  fi
  if ! grep -q 'runtime_selection_kernel_ready' \
    "docs/kaggle_cli_workflow.md"; then
    echo "error: Kaggle workflow doc does not describe runtime-selection kernel push readiness" >&2
    exit 1
  fi
  if ! grep -q 'runtime_selection_kernel_ready' \
    "docs/specs/README.md"; then
    echo "error: specs index does not describe runtime-selection kernel push readiness" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/eqvae-runtime-selection",
    "title": "eqvae runtime selection",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

expected_dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
if data.get("dataset_sources") != expected_dataset_sources:
    errors.append(f"dataset_sources must be exactly {expected_dataset_sources!r}")

for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for runtime selection")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_RUNTIME_SELECTION_READY = True" \
    --verify-only

  local run_file="$kernel_dir/run.py"
  python3 - "$run_file" <<'PY'
import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: runtime-selection run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = set(archive.namelist())
    errors: list[str] = []
    required_files = {
        "src/eqvae/benchmarking/runtime_selection.py",
        "src/eqvae/benchmarking/runtime_selection_executor.py",
        "src/eqvae/cli/runtime_selection_executor.py",
        "configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json",
        "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
        "runs/kaggle/runtime_selection_v5/benchmark/runtime_proof.json",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/runtime_proof.json",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/runtime_matrix.csv",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/dataloader_matrix.csv",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/numerical_checks.csv",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/corruption_checks.csv",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/gate_health_summary.json",
        "runs/kaggle/real_data_runtime_pretest_v8/metrics/gate_health.csv",
    }
    missing = sorted(required_files - names)
    if missing:
        errors.append(f"embedded payload missing required files: {missing!r}")
    unexpected_v8 = sorted(
        name for name in names
        if name.startswith("runs/kaggle/real_data_runtime_pretest_v8/")
        and name not in required_files
    )
    if unexpected_v8:
        errors.append(f"embedded payload has unexpected v8 files: {unexpected_v8!r}")
    try:
        executor_source = archive.read(
            "src/eqvae/benchmarking/runtime_selection_executor.py",
        ).decode("utf-8")
        writer_source = archive.read(
            "src/eqvae/benchmarking/runtime_selection.py",
        ).decode("utf-8")
        config = json.loads(
            archive.read(
                "configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json",
            ),
        )
        baseline = json.loads(
            archive.read(
                "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
            ),
        )
    except KeyError as error:
        print(f"error: runtime-selection payload missing {error}", file=sys.stderr)
        raise SystemExit(1) from None

    required_executor_text = (
        "RuntimeSelectionEvidence",
        "write_runtime_selection_benchmark",
        "--nproc_per_node=2",
        "torch.distributed.run",
        "DistributedDataParallel",
        "stain_corruptor_qa",
    )
    for text in required_executor_text:
        if text not in executor_source:
            errors.append(f"runtime-selection executor missing required text: {text}")
    required_writer_text = (
        "do_not_write_selected_runtime_if_missing_failed_or_skipped",
        "v8_hash_provenance_not_pass",
        "compiled_rows_diagnostic_only",
    )
    for text in required_writer_text:
        if text not in writer_source:
            errors.append(f"runtime-selection writer missing required text: {text}")
    runtime = config.get("runtime_matrix")
    if not isinstance(runtime, dict):
        errors.append("config.runtime_matrix must be an object")
    else:
        selection = runtime.get("selection_benchmark_slice")
        if not isinstance(selection, dict):
            errors.append("config.runtime_matrix.selection_benchmark_slice must be an object")
        elif selection.get("name") != "v8_shortlist_eager_amp_then_dual_gate":
            errors.append("selection slice must be v8_shortlist_eager_amp_then_dual_gate")
        else:
            efficiency = selection.get("efficiency_followup")
            if not isinstance(efficiency, dict):
                errors.append("selection efficiency_followup must be an object")
            else:
                expected_row = efficiency.get("baseline_row_id")
                expected_policy = efficiency.get("baseline_runtime_policy_id")
                snapshot = baseline.get("selected_row_snapshot")
                if not isinstance(snapshot, dict):
                    errors.append("baseline selected runtime must contain a snapshot")
                elif (
                    baseline.get("status") != "pass"
                    or baseline.get("selected_row_id") != expected_row
                    or baseline.get("runtime_policy_id") != expected_policy
                    or snapshot.get("row_id") != expected_row
                    or snapshot.get("runtime_policy_id") != expected_policy
                    or snapshot.get("status") != "pass"
                ):
                    errors.append("baseline selected runtime does not match config")
        carry = runtime.get("v8_carry_forward")
        if not isinstance(carry, dict):
            errors.append("config.runtime_matrix.v8_carry_forward must be an object")
        elif carry.get("full_run_eligible") is not False:
            errors.append("v8 carry-forward artifacts must remain non-promotable")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
PY

  for required_text in \
    "KAGGLE_RUNTIME_SELECTION_READY = True" \
    "v8_shortlist_eager_amp_then_dual_gate" \
    "runtime_selection_executor" \
    "selected_runtime.json" \
    "stain_corruptor_qa.json" \
    "runtime_proof.json" \
    "runtime_matrix.csv" \
    "dataloader_matrix.csv" \
    "numerical_checks.csv" \
    "corruption_checks.csv" \
    "gate_health_summary.json" \
    "single_visible_t4" \
    "dual_t4_ddp" \
    "torchrun" \
    "--nproc_per_node=2" \
    "wrong_accelerator"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: runtime-selection run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

guard_selected_runtime_lr_range_push_ready() {
  local kernel_dir="$1"
  local _metadata="$2"
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: LR-range kernel must be one generated run.py, not a sibling payload" >&2
    exit 1
  fi
  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOF'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the selected-runtime LR-range run.
EOF
    exit 1
  fi
  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin" >&2
    exit 1
  fi

  PYTHONPATH=src "$python_bin" - <<'PYLRCONFIG'
from pathlib import Path

from eqvae.config import resolve_json_config
from eqvae.training.selected_runtime import parse_selected_runtime_plan

plan = parse_selected_runtime_plan(
    Path("configs/spec0001/non_eq_vae_selected_runtime.json"),
)
config = resolve_json_config(
    Path("configs/spec0001/non_eq_vae_selected_runtime_lr_range.json"),
).effective_config
sweep = config.get("learning_rate_range")
training = config.get("training")
errors = []
if plan.per_device_batch_size != 25 or plan.global_batch_size != 50:
    errors.append("selected runtime must be the measured bs25/global50 winner")
if not isinstance(sweep, dict) or sweep.get("start") != 0.00002 \
        or sweep.get("end") != 0.003 or sweep.get("successful_updates") != 192:
    errors.append("LR range must be the bounded 2e-5..3e-3, 192-update sweep")
if not isinstance(training, dict) or training.get("max_train_steps") != 192:
    errors.append("LR range training.max_train_steps must be 192")
if errors:
    raise SystemExit("\n".join(f"error: {error}" for error in errors))
PYLRCONFIG

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True" \
    --verify-only \
    --allow-dirty
}

guard_selected_runtime_debug_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: selected-runtime debug gate must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOF'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the selected-runtime debug/tiny gate.
EOF
    exit 1
  fi

  if ! grep -q 'selected_runtime_debug_gate_contract_ready' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 does not describe the selected-runtime debug gate contract" >&2
    exit 1
  fi
  if ! grep -q 'selected_runtime_debug_gate_contract_ready' \
    "docs/specs/0003-kaggle-cli-execution-workflow.md"; then
    echo "error: spec 0003 does not describe the selected-runtime debug gate contract" >&2
    exit 1
  fi
  if ! grep -q 'selected_runtime_debug_gate_contract_ready' \
    "docs/kaggle_cli_workflow.md"; then
    echo "error: Kaggle workflow doc does not describe the selected-runtime debug gate contract" >&2
    exit 1
  fi
  if ! grep -q 'selected_runtime_debug_gate_contract_ready' \
    "docs/specs/README.md"; then
    echo "error: specs index does not describe the selected-runtime debug gate contract" >&2
    exit 1
  fi

  local python_bin="${PYTHON:-.venv/bin/python}"
  if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
  fi
  PYTHONPATH=src "$python_bin" -m eqvae.cli.selected_runtime_gate \
    --verify-push-ready \
    --selector-generation-mode remote_generate \
    --debug-config configs/spec0001/non_eq_vae_selected_runtime_debug.json \
    --tiny-config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
    --runtime-config configs/spec0001/non_eq_vae_selected_runtime.json \
    --fixed-train-patches configs/spec0001/fixed_32_train_overfit_patches.json
  preflight_fixed32_selector_readiness

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/eqvae-selected-runtime-debug",
    "title": "eqvae selected runtime debug",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

expected_dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
if data.get("dataset_sources") != expected_dataset_sources:
    errors.append(f"dataset_sources must be exactly {expected_dataset_sources!r}")

for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for selected-runtime debug")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True" \
    --verify-only \
    --allow-dirty

  local run_file="$kernel_dir/run.py"
  PYTHONPATH=src "$python_bin" - "$run_file" <<'PYDEBUGPAYLOAD'
import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from eqvae.training.selected_runtime import selected_runtime_plan_errors

run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: selected-runtime debug run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = set(archive.namelist())
    errors: list[str] = []
    required_run_text = (
        "selector_generation.get(\"status\") == \"pass\"",
        "_generate_remote_fixed32_selector(",
        "_fixed32_selector_status_from_payload_cwd(",
        "fixed32_selector_status(selector_path, data_root=data_root)",
        "_run_real_selected_runtime_debug(",
        "_run_real_selected_runtime_tiny_overfit(",
        "_write_real_gate_summary(",
        "_run_selected_runtime_train_torchrun(",
        "_selected_runtime_train_torchrun_command(",
        "\"torch.distributed.run\"",
        "\"--standalone\"",
        "\"--nproc_per_node=2\"",
        "\"eqvae.cli.selected_runtime_train\"",
        "\"--fixed-train-patches\"",
        "DEBUG_RESUME_STEP = 4",
        "DEBUG_FINAL_STEP = 8",
        "TINY_MAX_STEP = 128",
        "\"--resume\"",
        "step_{DEBUG_RESUME_STEP:06d}.pt",
        "_validate_real_runner_artifacts(output_dir=output_dir)",
    )
    for text in required_run_text:
        if text not in run_text:
            errors.append(f"selected-runtime debug run.py missing required source text: {text}")
    required_files = {
        "src/eqvae/benchmarking/fixed32_selector_readiness.py",
        "src/eqvae/benchmarking/selected_runtime_gate.py",
        "src/eqvae/cli/fixed32_selector_readiness.py",
        "src/eqvae/cli/select_fixed_patches.py",
        "src/eqvae/cli/selected_runtime_gate.py",
        "src/eqvae/cli/selected_runtime_train.py",
        "src/eqvae/cli/train.py",
        "src/eqvae/training/debug.py",
        "src/eqvae/training/selected_runtime_runner.py",
        "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
        "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
        "configs/spec0001/fixed_32_train_overfit_patches.json",
        "configs/spec0001/non_eq_vae_selected_runtime.json",
        "configs/spec0001/non_eq_vae_runtime_winner.json",
    }
    missing = sorted(required_files - names)
    if missing:
        errors.append(f"embedded payload missing required files: {missing!r}")
    try:
        gate_source = archive.read(
            "src/eqvae/benchmarking/selected_runtime_gate.py",
        ).decode("utf-8")
        runner_source = archive.read(
            "src/eqvae/training/selected_runtime_runner.py",
        ).decode("utf-8")
        debug_config = json.loads(
            archive.read("configs/spec0001/non_eq_vae_selected_runtime_debug.json"),
        )
        tiny_config = json.loads(
            archive.read("configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json"),
        )
        fixed_selector = json.loads(
            archive.read("configs/spec0001/fixed_32_train_overfit_patches.json"),
        )
        selected_runtime = json.loads(
            archive.read("configs/spec0001/non_eq_vae_selected_runtime.json"),
        )
    except KeyError as error:
        print(f"error: selected-runtime debug payload missing {error}", file=sys.stderr)
        raise SystemExit(1) from None

    # Spec 0011 S17b-3: delegate the selected-runtime identity/recipe/snapshot/batch
    # validation to the single-source parser instead of mirroring its frozen eager
    # literals here. This accepts a re-measured compiled winner (amp-off whole-step
    # compile, any exact batch) while keeping every hardware/topology anchor -- the
    # parser pins accelerator/machine_shape/world_size/nproc/grad-accum even more
    # explicitly than the old identity literal did (which carried them only
    # incidentally). selected_runtime_path is None: the launch parse re-checks the
    # runtime proof; this push guard only needs the proof file present (required_files
    # above). Byte-identical acceptance on the committed v5 plan.
    if not isinstance(selected_runtime, dict):
        errors.append("selected runtime must be a JSON object")
    else:
        errors.extend(
            selected_runtime_plan_errors(
                selected_runtime,
                selected_runtime_path=None,
            ),
        )

    debug_gate = debug_config.get("selected_runtime_debug")
    tiny_gate = tiny_config.get("selected_runtime_debug_gate")
    for name, gate in (
        ("selected_runtime_debug", debug_gate),
        ("selected_runtime_debug_gate", tiny_gate),
    ):
        if not isinstance(gate, dict):
            errors.append(f"{name} must be an object")
            continue
        expected_gate_values = {
            "remote_pass_ready": False,
            "real_train_runner_implemented": True,
            "selector_generation_mode": "remote_generate",
            "remote_selector_generation_ready": True,
            "fixed_32_selector_real": False,
        }
        for key, expected in expected_gate_values.items():
            if gate.get(key) != expected:
                errors.append(f"{name}.{key} must be {expected!r} before remote push")

    stale_blockers = (
        "real_ubc_selected_runtime_train_runner_not_implemented",
        "selected_runtime_debug_wrapper_not_wired_to_real_runner_until_spec0008",
    )
    for stale in stale_blockers:
        if stale in gate_source:
            errors.append(f"selected-runtime debug gate contains stale blocker {stale}")
    if "Only data='synthetic' is implemented" in runner_source:
        errors.append("train runner still rejects data='ubc-pre-shuffled'")
    required_runner_text = (
        "_loaded_checkpoint_resume_proof",
        "loaded_successful_optimizer_update_count",
        "additional_optimizer_steps",
        "ubc-pre-shuffled",
    )
    for text in required_runner_text:
        if text not in runner_source:
            errors.append(f"selected-runtime runner missing required source text: {text}")
    if fixed_selector.get("status") != "requires_real_data_generation":
        errors.append("embedded fixed_32 selector should remain a remote-generate placeholder")
    selectors = fixed_selector.get("selectors")
    if not isinstance(selectors, list) or selectors:
        errors.append("embedded fixed_32 placeholder must not contain local selectors")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
PYDEBUGPAYLOAD

  for required_text in \
    "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True" \
    "selected_runtime_debug_gate_contract_ready" \
    "remote_generate" \
    "select_fixed_patches" \
    "fixed_32_train_overfit" \
    "fixed32_selector_readiness" \
    "selected_runtime_gate" \
    "selected_runtime_train" \
    "selected_runtime_gate_summary.json" \
    "selected_runtime_debug_summary.json" \
    "selected_runtime_plan_applied.json" \
    "local_selected_runtime_readiness.json" \
    "checkpoint_resume_proof.json" \
    "tiny_overfit_summary.json" \
    "artifact_manifest.json" \
    "gate_health_summary.json" \
    "selected_runtime.json" \
    "single_visible_t4" \
    "dual_t4_ddp" \
    "torchrun" \
    "--nproc_per_node=2" \
    "wrong_accelerator"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: selected-runtime debug run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

guard_selected_runtime_full_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  local guard_mode="${3:-push}"

  case "$guard_mode" in
    push|local_preflight)
      ;;
    *)
      echo "error: unknown selected-runtime full guard mode: $guard_mode" >&2
      exit 1
      ;;
  esac

  if [[ "$guard_mode" != "local_preflight" ]] && \
    [[ "${EQVAE_SELECTED_RUNTIME_FULL_LOCAL_PREFLIGHT_ALLOW_DIRTY:-}" == "1" ]]; then
    cat >&2 <<'EOFGUARD'
error: EQVAE_SELECTED_RUNTIME_FULL_LOCAL_PREFLIGHT_ALLOW_DIRTY is only valid
inside preflight-selected-runtime-full; unset it before a real push guard.
EOFGUARD
    exit 1
  fi

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: selected-runtime full run must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOFGUARD'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the selected-runtime full training run.
EOFGUARD
    exit 1
  fi

  if ! grep -q 'selected_runtime_full_run_contract_ready' \
    "configs/spec0001/non_eq_vae_selected_runtime_full.json"; then
    echo "error: full config does not carry the selected-runtime full contract token" >&2
    exit 1
  fi
  if ! grep -q 'selected_runtime_full_run_contract_ready' \
    "$kernel_dir/run_template.py"; then
    echo "error: full kernel template does not carry the selected-runtime full contract token" >&2
    exit 1
  fi
  if ! grep -q '0009-first-full-selected-runtime-training-run.md' \
    "docs/specs/README.md"; then
    echo "error: specs index does not list spec 0009" >&2
    exit 1
  fi

  local python_bin="${PYTHON:-.venv/bin/python}"
  if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
  fi
  PYTHONPATH=src "$python_bin" -m eqvae.cli.selected_runtime_gate \
    --verify-output \
    --output-dir runs/kaggle/selected_runtime_debug \
    --runtime-config configs/spec0001/non_eq_vae_selected_runtime.json

  python3 - "$metadata" <<'PYFULLMETA'
import json
import sys
from pathlib import Path
metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []
required = {
    "id": "maximusshtefan/eqvae-selected-runtime-full",
    "title": "eqvae selected runtime full",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}
for key, expected in required.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")
expected_datasets = [
    "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    "maximusshtefan/eqvae-baseline-session2-step45000",
]
if data.get("dataset_sources") != expected_datasets:
    errors.append("dataset_sources must attach the exact UBC and session-2 datasets")
for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be empty")
if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PYFULLMETA

  local verify_args=(
    --kernel-dir "$kernel_dir"
    --ready-marker "KAGGLE_SELECTED_RUNTIME_FULL_READY = True"
    --verify-only
  )
  if [[ "$guard_mode" == "local_preflight" ]]; then
    verify_args+=(--allow-dirty)
  fi
  build_kernel_py "${verify_args[@]}"

  local run_file="$kernel_dir/run.py"
  PYTHONPATH=src "$python_bin" - "$run_file" <<'PYFULLPAYLOAD'
import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from eqvae.benchmarking.schedule import training_steps_per_epoch
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT


def _positive_int_or_none(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _int_or_none(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""', run_text, flags=re.DOTALL)
if match is None:
    print("error: selected-runtime full run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)
payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = set(archive.namelist())
    errors: list[str] = []
    required_files = {
        "src/eqvae/benchmarking/selected_runtime_gate.py",
        "src/eqvae/cli/selected_runtime_gate.py",
        "src/eqvae/cli/selected_runtime_train.py",
        "src/eqvae/training/selected_runtime.py",
        "src/eqvae/training/selected_runtime_runner.py",
        "configs/spec0001/non_eq_vae_selected_runtime_full.json",
        "configs/spec0001/non_eq_vae_selected_runtime.json",
        "configs/spec0001/non_eq_vae_runtime_winner.json",
    }
    missing = sorted(required_files - names)
    if missing:
        errors.append(f"embedded payload missing required files: {missing!r}")
    full_config = json.loads(archive.read("configs/spec0001/non_eq_vae_selected_runtime_full.json"))
    selected_runtime = json.loads(archive.read("configs/spec0001/non_eq_vae_selected_runtime.json"))
    training = full_config.get("training") if isinstance(full_config, dict) else None
    objective = full_config.get("objective") if isinstance(full_config, dict) else None
    beta = objective.get("beta") if isinstance(objective, dict) else None
    if not isinstance(beta, dict) or beta.get("target") != 0.01:
        errors.append("full config objective.beta.target must be locked to 0.01")
    # Spec 0011 S8: derive the schedule from the plan's measured global batch and the
    # single-sourced patch count instead of pinning the reference literals. At the
    # reference global batch 24 these reproduce 12500/125000/6250 exactly, so the built
    # kernel is unchanged; a re-measured non-24 plan is validated by relationship.
    per_device_batch = _positive_int_or_none(selected_runtime.get("per_device_batch_size"))
    world_size = _positive_int_or_none(selected_runtime.get("world_size"))
    global_batch = _positive_int_or_none(selected_runtime.get("global_batch_size"))
    epochs = _int_or_none(training.get("epochs")) if isinstance(training, dict) else None
    if isinstance(training, dict) and "epochs" in training and epochs is None:
        # A present-but-non-int epochs (e.g. JSON 10.0) must fail closed here: it would
        # otherwise pass the ``!= 10`` anchor pin yet null the derivation, silently
        # skipping the FULL_TARGET_UPDATES/FULL_HALF_EPOCH_INTERVAL run.py token check.
        errors.append("full config training.epochs must be an integer")
    if global_batch is None:
        errors.append("selected runtime global_batch_size must be a positive integer")
        derived_updates = None
    else:
        derived_updates = training_steps_per_epoch(
            real_train_patch_count=REAL_TRAIN_PATCH_COUNT,
            global_batch_size=global_batch,
        )
    if global_batch is not None and (
        per_device_batch is None
        or world_size is None
        or global_batch != per_device_batch * world_size
    ):
        errors.append(
            "selected runtime global_batch_size must equal "
            "per_device_batch_size * world_size"
        )
    if derived_updates is None or epochs is None:
        derived_target = None
        derived_half = None
    else:
        derived_target = epochs * derived_updates
        derived_half = derived_updates // 2
    if not isinstance(training, dict):
        errors.append("full config training must be an object")
    else:
        expected_training = {
            "epochs": 10,
            "train_reparameterization": "stochastic_seeded",
            "checkpoint_retention": "best_final_latest_four_interval",
            "resume_supported": True,
        }
        for key, expected in expected_training.items():
            if training.get(key) != expected:
                errors.append(f"full config training.{key} must be {expected!r}")
        for key in (
            "optimizer_updates_per_epoch",
            "max_train_steps",
            "half_epoch_interval_steps",
            "save_every_steps",
        ):
            if key in training:
                errors.append(
                    f"full config training must not re-freeze {key}; "
                    "schedule is runner-derived (Spec 0011)"
                )
    recorded_updates = selected_runtime.get("optimizer_updates_per_epoch")
    if derived_updates is not None and (
        not isinstance(recorded_updates, int)
        or isinstance(recorded_updates, bool)
        or recorded_updates != derived_updates
    ):
        errors.append(
            f"selected runtime optimizer_updates_per_epoch must be {derived_updates!r}"
        )
    forbidden = (
        "selected_runtime_debug",
        "DEBUG_FINAL_STEP",
        "TINY_MAX_STEP",
        "non_eq_vae_selected_runtime_debug.json",
    )
    required_text = (
        "KAGGLE_SELECTED_RUNTIME_FULL_READY = True",
        "selected_runtime_full_run_contract_ready",
        "non_eq_vae_selected_runtime_full.json",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "eqvae.cli.selected_runtime_train",
        "--resume",
        "maximusshtefan/eqvae-baseline-session2-step45000",
        "/kaggle/input/eqvae-baseline-session2-step45000/step_045000.pt",
        "e7a0f05e013bff4f7a5bfbfd4442f3c9a6d19cf261c42f54a6d04391be76e88b",
        "dual_t4_ddp",
    )
    for token in required_text:
        if token not in run_text:
            errors.append(f"full run.py missing required text: {token}")
    if derived_target is not None and derived_half is not None:
        for token in (
            f"FULL_TARGET_UPDATES = {derived_target}",
            f"FULL_HALF_EPOCH_INTERVAL = {derived_half}",
        ):
            if token not in run_text:
                errors.append(f"full run.py missing required text: {token}")
    forbidden_command_token = '"--max-train-steps"'
    command_builder = "_selected_runtime_full_torchrun_command"
    command_block_match = re.search(
        rf"def {command_builder}\(.*?(?=\ndef _)",
        run_text,
        flags=re.DOTALL,
    )
    if command_block_match is None:
        errors.append("full run.py missing selected-runtime full torchrun builder")
    else:
        command_block = command_block_match.group(0)
        for token in forbidden:
            if token in command_block:
                errors.append(f"full torchrun command contains debug/tiny token: {token}")
        if forbidden_command_token in command_block:
            errors.append("full torchrun command must not contain --max-train-steps")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
PYFULLPAYLOAD
}


validate_payload_freshness() {
  local payload_dir="$1"
  python3 - "$payload_dir" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

payload = Path(sys.argv[1])
manifest_path = payload / "payload_manifest.json"
if not manifest_path.exists():
    print("error: missing payload_manifest.json; rebuild kernel payload", file=sys.stderr)
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
errors: list[str] = []


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_tree(path: Path) -> str:
    hasher = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(digest_file(item).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if manifest.get("schema_version") != "spec0001.kaggle_payload_manifest.v1":
    errors.append("payload manifest has an unexpected schema_version")
if manifest.get("git_dirty") is not False:
    errors.append("payload was built from a dirty git worktree")
if manifest.get("git_commit") != git_output("rev-parse", "HEAD"):
    errors.append("payload git_commit does not match current HEAD; rebuild payload")

entries = manifest.get("entries")
expected_entries = {
    "src/eqvae": digest_tree(Path("src/eqvae")),
    "configs/spec0001": digest_tree(Path("configs/spec0001")),
    "docs/data/ubc_ocean_masked_holdout_ids.csv": digest_file(
        Path("docs/data/ubc_ocean_masked_holdout_ids.csv")
    ),
    "pyproject.toml": digest_file(Path("pyproject.toml")),
    "uv.lock": digest_file(Path("uv.lock")),
}
if not isinstance(entries, dict):
    errors.append("payload manifest entries must be an object")
else:
    for key, expected in expected_entries.items():
        if entries.get(key) != expected:
            errors.append(f"payload entry {key!r} is stale; rebuild payload")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

guard_clean_kernel_dir() {
  local kernel_dir="${1:-$default_kernel_dir}"
  if [[ -n "$(git status --short -- "$kernel_dir")" ]]; then
    cat >&2 <<'EOF'
error: local kernel directory has uncommitted changes.

Commit/stash/reconcile local changes before pulling from Kaggle, or pull into a
separate temporary directory manually.
EOF
    exit 1
  fi
}

api_check() {
  local kernel_dir="${1:-$default_kernel_dir}"
  require_remote_confirmed
  require_kaggle_cli
  local actor
  local kernel_id
  local kernel_exists
  local original_kernel_id
  local kernel_listing
  local source_rows
  actor="$(kaggle_authenticated_username)"
  original_kernel_id="$(kernel_id_from_metadata "$kernel_dir")"
  kernel_id="${actor}/${original_kernel_id#*/}"

  echo "Kaggle API read-only preflight"
  echo "=============================="
  kaggle --version
  kaggle_auth_path_message
  echo "ok: authenticated Kaggle actor is $actor"

  kernel_listing="$(
    kaggle_api kernels list --mine --search "${kernel_id#*/}" --csv
  )"
  echo "ok: kernels list works for search ${kernel_id#*/}"

  kernel_exists="$(
    KAGGLE_KERNEL_LISTING="$kernel_listing" python3 - "$kernel_id" <<'PYKERNELLIST'
import csv
import io
import os
import sys

kernel_id = sys.argv[1]
rows = csv.reader(io.StringIO(os.environ["KAGGLE_KERNEL_LISTING"]))
print("1" if any(kernel_id in row for row in rows) else "0")
PYKERNELLIST
  )"
  if [[ "$kernel_exists" == "1" ]]; then
    kaggle_api kernels status "$kernel_id" >/dev/null
    echo "ok: kernels status works for existing $kernel_id"
    kaggle_api kernels logs "$kernel_id" >/dev/null
    echo "ok: kernels logs works for existing $kernel_id"
  else
    echo "ok: $kernel_id has no existing version; first launch may proceed"
  fi

  source_rows="$(
    python3 - "$kernel_dir/kernel-metadata.json" <<'PYKAGGLESOURCES'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for field in (
    "dataset_sources",
    "competition_sources",
    "kernel_sources",
    "model_sources",
):
    sources = metadata.get(field)
    if not isinstance(sources, list):
        raise SystemExit(f"{field} must be a list")
    for source in sources:
        if not isinstance(source, str) or not source or "\t" in source or "\n" in source:
            raise SystemExit(f"{field} contains an invalid source locator")
        print(f"{field}\t{source}")
PYKAGGLESOURCES
  )"
  while [[ -n "$source_rows" ]] \
    && IFS=$'\t' read -r source_kind source_reference; do
    case "$source_kind" in
    dataset_sources)
      kaggle_api datasets files "$source_reference" -v >/dev/null
      ;;
    competition_sources)
      kaggle_api competitions files "$source_reference" -v >/dev/null
      ;;
    kernel_sources)
      kaggle_api kernels files "$source_reference" -v >/dev/null
      ;;
    model_sources)
      kaggle_api models instances versions files "$source_reference" -v >/dev/null
      ;;
    *)
      echo "error: unsupported Kaggle source kind: $source_kind" >&2
      exit 1
      ;;
    esac
    echo "ok: $source_kind is readable at $source_reference"
  done <<<"$source_rows"

  if kaggle_api quota -v >/dev/null 2>&1; then
    echo "ok: accelerator quota endpoint works"
  else
    echo "warn: accelerator quota endpoint failed; verify quota in Kaggle UI before remote benchmark push" >&2
  fi

  if [[ "$kernel_exists" != "1" ]]; then
    echo "ok: kernels files skipped before first launch"
  elif kaggle_api kernels files "$kernel_id" -v >/dev/null 2>&1; then
    echo "ok: kernels files endpoint works for existing $kernel_id"
  else
    echo "warn: kernels files endpoint failed; status/logs still work, but source-file introspection is unavailable" >&2
  fi
}

preflight_runtime_selection() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  build_embedded_kernel "$runtime_selection_kernel_dir"
  validate_kernel_dir "$runtime_selection_kernel_dir"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_runtime_selection_benchmark.py \
    tests/test_kaggle_embedded_kernel.py::test_embedded_runtime_selection_kernel_import_simulation \
    tests/test_kaggle_embedded_kernel.py::test_embedded_runtime_selection_kernel_full_local_fail_closed_simulation \
    -q
}

preflight_fixed25_selector() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  build_embedded_kernel "$fixed25_selector_kernel_dir"
  validate_kernel_dir "$fixed25_selector_kernel_dir"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_fixed_selectors.py \
    tests/test_fixed25_equivariance_artifacts.py \
    tests/test_kaggle_embedded_kernel.py::test_embedded_fixed25_selector_kernel_import_simulation \
    -q
}

preflight_fixed32_selector_readiness() {
  local python_bin="${PYTHON:-.venv/bin/python}"
  local synthetic_root="/tmp/eqvae-fixed32-synthetic-root"
  local output_dir="$synthetic_root/readiness"
  local selector_output="$synthetic_root/fixed_32_train_overfit_patches.json"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_fixed_selectors.py \
    tests/test_fixed32_selector_readiness.py \
    -q

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m eqvae.cli.fixed32_selector_readiness \
    --config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
    --synthetic-root "$synthetic_root" \
    --output-dir "$output_dir" \
    --masked-holdout-csv docs/data/ubc_ocean_masked_holdout_ids.csv

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m eqvae.cli.select_fixed_patches \
    --config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
    --kind fixed_32_train_overfit \
    --data-root "$synthetic_root" \
    --masked-holdout-csv docs/data/ubc_ocean_masked_holdout_ids.csv \
    --output "$selector_output" \
    --validate-crc

  "$python_bin" - "$output_dir" "$selector_output" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
selector_output = Path(sys.argv[2])
readiness_path = output_dir / "benchmark" / "fixed32_selector_readiness.json"
errors = []
if not selector_output.exists():
    errors.append(f"selector output missing: {selector_output}")
if not readiness_path.exists():
    errors.append(f"readiness artifact missing: {readiness_path}")
else:
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    synthetic_status = readiness.get("synthetic_selector_status")
    if not isinstance(synthetic_status, dict):
        errors.append("synthetic_selector_status must be an object")
        synthetic_status = {}
    expected = {
        "status": "pass",
        "selector_generation_mode": "remote_generate",
        "remote_selector_generation_ready": True,
        "fixed_32_selector_real": False,
        "synthetic_selector_deterministic": True,
        "synthetic_selector_canonical_real_rejected": True,
    }
    for key, value in expected.items():
        if readiness.get(key) != value:
            errors.append(f"{key} must be {value!r}")
    if synthetic_status.get("failure_kind") != "fixed_32_selector_not_canonical_real_ubc":
        errors.append("synthetic selector must fail canonical-real readiness")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

preflight_selected_runtime_debug() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  preflight_fixed32_selector_readiness
  build_embedded_kernel "$selected_runtime_debug_kernel_dir"
  validate_kernel_dir "$selected_runtime_debug_kernel_dir"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_selected_runtime_gate.py \
    tests/test_kaggle_embedded_kernel.py::test_embedded_selected_runtime_debug_kernel_import_simulation \
    tests/test_kaggle_embedded_kernel.py::test_embedded_selected_runtime_debug_kernel_full_local_fail_closed_simulation \
    -q
}

preflight_selected_runtime_lr_range() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi
  build_embedded_kernel "$selected_runtime_lr_range_kernel_dir"
  validate_kernel_dir "$selected_runtime_lr_range_kernel_dir"
  KAGGLE_FULL_DATASET_CONFIRMED=1 guard_selected_runtime_lr_range_push_ready \
    "$selected_runtime_lr_range_kernel_dir" \
    "$selected_runtime_lr_range_kernel_dir/kernel-metadata.json"
  "$python_bin" -m pytest -q tests/test_selected_runtime_full_run.py \
    -k 'spec0011_checked_in_winner_plan or spec0011_winner_plan_rejects or lr_range'
}

preflight_selected_runtime_full() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  PYTHONPATH=src "$python_bin" -m eqvae.cli.selected_runtime_gate \
    --verify-output \
    --output-dir runs/kaggle/selected_runtime_debug \
    --runtime-config configs/spec0001/non_eq_vae_selected_runtime.json

  build_embedded_kernel "$selected_runtime_full_kernel_dir"
  validate_kernel_dir "$selected_runtime_full_kernel_dir"
  EQVAE_SELECTED_RUNTIME_FULL_LOCAL_PREFLIGHT_ALLOW_DIRTY=1 \
    KAGGLE_FULL_DATASET_CONFIRMED=1 guard_selected_runtime_full_push_ready \
    "$selected_runtime_full_kernel_dir" \
    "$selected_runtime_full_kernel_dir/kernel-metadata.json" \
    "local_preflight"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_selected_runtime_full_run.py \
    tests/test_kaggle_embedded_kernel.py::test_embedded_selected_runtime_full_kernel_import_simulation \
    -q
}


preflight_selected_runtime_runner() {
  local python_bin="${PYTHON:-.venv/bin/python}"
  local output_dir="$TMPDIR/selected_runtime_runner_preflight_$$"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi
  rm -rf "$output_dir"

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_selected_runtime_gate.py \
    tests/test_train_cli.py \
    tests/test_selected_runtime_runner.py \
    -q

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m eqvae.cli.selected_runtime_train \
    --config configs/spec0001/non_eq_vae_selected_runtime_debug.json \
    --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json \
    --data synthetic \
    --output-dir "$output_dir" \
    --run-name spec0007_local_runner_dryrun \
    --max-train-steps 2 \
    --max-val-steps 1 \
    --dry-run

  "$python_bin" - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
benchmark = output_dir / "benchmark"
metrics = output_dir / "metrics"
required = [
    benchmark / "training_summary.json",
    benchmark / "selected_runtime_debug_summary.json",
    benchmark / "selected_runtime_plan_applied.json",
    benchmark / "checkpoint_resume_proof.json",
    benchmark / "gate_health_summary.json",
    benchmark / "artifact_manifest.json",
    metrics / "train_steps.csv",
    metrics / "gate_health.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print(f"error: selected-runtime runner preflight missing artifacts: {missing}", file=sys.stderr)
    raise SystemExit(1)

summary = json.loads((benchmark / "training_summary.json").read_text(encoding="utf-8"))
debug = json.loads((benchmark / "selected_runtime_debug_summary.json").read_text(encoding="utf-8"))
plan = json.loads((benchmark / "selected_runtime_plan_applied.json").read_text(encoding="utf-8"))
manifest = json.loads((benchmark / "artifact_manifest.json").read_text(encoding="utf-8"))
readiness = json.loads((benchmark / "local_selected_runtime_readiness.json").read_text(encoding="utf-8"))

errors = []
if summary.get("full_run_eligible") is not False:
    errors.append("training summary must remain non-promotable")
if debug.get("real_train_runner_implemented") is not True:
    errors.append("runner readiness must prove real_train_runner_implemented=true")
if debug.get("remote_pass_ready") is not False:
    errors.append("runner dry-run must not claim remote_pass_ready")
if plan.get("status") != "fail" or plan.get("plan_applied") is not False:
    errors.append("local dry-run must fail full dual-T4/AMP plan application")
if readiness.get("remote_pass_ready") is not False:
    errors.append("local readiness must keep remote_pass_ready=false")
if manifest.get("status") != "local_pass":
    errors.append("artifact manifest must pass locally")
if (benchmark / "selected_runtime.json").exists():
    errors.append("runner preflight must not write benchmark/selected_runtime.json")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
  rm -rf "$output_dir"
}

build_latent_inference() {
  local mode="${1:-}"
  local run_number="${2:-}"
  if [[ "$mode" != "pilot" && "$mode" != "production-all" && "$mode" != "resume" && "$mode" != "finalizer" ]]; then
    echo "error: build-latent-inference requires pilot, production-all, finalizer, or resume XX" >&2
    exit 1
  fi
  if [[ "$mode" == "resume" ]]; then
    run_number="$(normalize_latent_run_number "$run_number")"
  fi
  require_build_python
  local build_args=(
    -m eqvae.cli.build_ubc_latent_kernel "$mode"
    --repo-root "$PWD"
    --output-root "$latent_inference_kernel_root"
    --input-contract "$latent_input_bundle_dir/spec0021_input_contract.json"
    --input-receipt "$latent_input_receipt"
  )
  if [[ "$mode" == "resume" ]]; then
    build_args+=(
      --run-number "$((10#$run_number))"
      --resume-receipt \
        "$latent_input_authority_dir/resume_run_${run_number}_dataset_receipt.json"
    )
  fi
  "$build_python" "${build_args[@]}"
  if [[ "$mode" == "resume" ]]; then
    preflight_latent_inference "run-$run_number"
  else
    preflight_latent_inference "$mode"
  fi
}

normalize_latent_run_number() {
  local value="${1:-}"
  if [[ ! "$value" =~ ^(0[1-5]|[1-5])$ ]]; then
    echo "error: latent run number must be 01, 02, 03, 04, or 05" >&2
    exit 1
  fi
  printf '%02d\n' "$((10#$value))"
}

validate_latent_kernel_dir() {
  local kernel_dir="$1"
  local receipt_policy="${2:-allow-null}"
  python3 - \
    "$kernel_dir" \
    "$receipt_policy" \
    "$latent_input_dataset_slug" \
    "$latent_input_bundle_dir/spec0021_input_contract.json" \
    "docs/specs/0021-dual-model-wsi-latent-inference.md" \
    "$latent_input_authority_dir" <<'PYLATENT'
import hashlib
import json
import sys
from pathlib import Path

kernel_dir = Path(sys.argv[1])
receipt_policy = sys.argv[2]
dataset_slug = sys.argv[3]
input_contract_path = Path(sys.argv[4])
spec_path = Path(sys.argv[5])
authority_dir = Path(sys.argv[6])
expected_files = {
    "kernel-metadata.json",
    "run.py",
    "spec0021_inference_config.json",
}
observed_files = {
    path.relative_to(kernel_dir).as_posix()
    for path in kernel_dir.rglob("*")
    if path.is_file()
}
errors: list[str] = []
if observed_files != expected_files:
    errors.append(
        f"upload directory files differ: {sorted(observed_files)!r}"
    )


def read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


metadata = read_object(kernel_dir / "kernel-metadata.json")
config_path = kernel_dir / "spec0021_inference_config.json"
config = read_object(config_path)
canonical_config = json.dumps(
    config,
    sort_keys=True,
    separators=(",", ":"),
) + "\n"
if config_path.read_text(encoding="utf-8") != canonical_config:
    errors.append("inference config is not canonical compact JSON")

config_fields = {
    "schema_version",
    "mode",
    "run_number",
    "spec_sha256",
    "input_dataset_receipt",
    "input_contract_sha256",
    "work_manifest_sha256",
    "normal_checkpoint_sha256",
    "so2_checkpoint_sha256",
    "pilot_authority_sha256",
    "selected_recipe",
    "expected_binary_output_bytes",
    "saved_output_limit_bytes",
    "resume_dataset_receipt",
}
if set(config) != config_fields:
    errors.append("inference config fields differ from the locked schema")
if config.get("schema_version") != "spec0021.inference_config.v2":
    errors.append("inference config schema is not Spec 0021")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if config.get("spec_sha256") != sha256(spec_path):
    errors.append("inference config Spec 0021 hash is stale")
if config.get("input_contract_sha256") != sha256(input_contract_path):
    errors.append("inference config input-contract hash is stale")
expected_checkpoints = {
    "normal_checkpoint_sha256": (
        "f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075"
    ),
    "so2_checkpoint_sha256": (
        "041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7"
    ),
}
for key, expected in expected_checkpoints.items():
    if config.get(key) != expected:
        errors.append(f"{key} is not the frozen checkpoint hash")

receipt = config.get("input_dataset_receipt")
dataset_sources: list[str] = []
if receipt is None:
    if receipt_policy != "allow-null":
        errors.append("push requires a pinned input dataset receipt")
elif not isinstance(receipt, dict):
    errors.append("input dataset receipt must be an object")
else:
    if receipt.get("schema_version") != "spec0021.input_dataset_receipt.v1":
        errors.append("input dataset receipt schema mismatch")
    reference = receipt.get("dataset_reference")
    if reference != dataset_slug:
        errors.append("input dataset receipt reference is not the fixed dataset")
    else:
        dataset_sources.append(reference)
    input_receipt_path = authority_dir / "input_dataset_receipt.json"
    if not input_receipt_path.is_file() or receipt != read_object(input_receipt_path):
        errors.append("embedded input receipt differs from local authority")

mode = config.get("mode")
run_number = config.get("run_number")
work_hashes = {
    1: "76cc5f9b86b75b9e46250c80e7b5c98d2f0451a12b38665ae45b055767b7a456",
    2: "11d21482c9d3e083bc5138973c6b09f6b659e7d753853c0290382320e78e6200",
    3: "9414f638abc24963e821de8bbedccaf294a14a7ea768a01687d5b105e634402b",
    4: "e7c5d8d08996e3bac440b5779547b2bbeb4443e7481dcb74998a87aa3054697f",
    5: "5485c44ababeaba9a4e2cc75a1a6b2927d89f10ed83a121dfcb81a68cfc23d6a",
}
if mode == "pilot":
    expected_id = "maximusshtefan/eqvae-ubc-ocean-latent-pilot"
    if run_number is not None or config.get("work_manifest_sha256") is not None:
        errors.append("pilot config must not bind a production run")
    for key in (
        "pilot_authority_sha256",
        "selected_recipe",
        "expected_binary_output_bytes",
        "saved_output_limit_bytes",
        "resume_dataset_receipt",
    ):
        if config.get(key) is not None:
            errors.append(f"pilot config field {key} must be null")
elif mode == "production" and isinstance(run_number, int) and run_number in work_hashes:
    expected_id = f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run_number:02d}"
    if config.get("work_manifest_sha256") != work_hashes[run_number]:
        errors.append("production work-manifest hash is stale")
    resume_receipt = config.get("resume_dataset_receipt")
    if receipt_policy == "fresh" and resume_receipt is not None:
        errors.append("fresh production config must not attach resume data")
    elif receipt_policy == "resume":
        expected_resume_reference = (
            f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run_number:02d}-resume"
        )
        if not isinstance(resume_receipt, dict):
            errors.append("resume production requires its pinned dataset receipt")
        else:
            if resume_receipt.get("schema_version") != (
                "spec0021.resume_dataset_receipt.v1"
            ):
                errors.append("resume dataset receipt schema mismatch")
            if resume_receipt.get("dataset_reference") != expected_resume_reference:
                errors.append("resume dataset receipt reference/run mismatch")
            if resume_receipt.get("run_number") != run_number:
                errors.append("resume dataset receipt run mismatch")
            if resume_receipt.get("input_bundle_sha256") != sha256(input_contract_path):
                errors.append("resume dataset receipt input contract mismatch")
            if resume_receipt.get("work_manifest_sha256") != work_hashes[run_number]:
                errors.append("resume dataset receipt work manifest mismatch")
            resume_receipt_path = (
                authority_dir
                / f"resume_run_{run_number:02d}_dataset_receipt.json"
            )
            if (
                not resume_receipt_path.is_file()
                or resume_receipt != read_object(resume_receipt_path)
            ):
                errors.append("embedded resume receipt differs from local authority")
            dataset_sources.append(expected_resume_reference)
    elif receipt_policy not in {"fresh", "resume"}:
        errors.append("production preflight must declare fresh or resume policy")
    for key in (
        "pilot_authority_sha256",
        "selected_recipe",
    ):
        if config.get(key) is None:
            errors.append(f"production config field {key} must be bound")
    fixed_recipe = {
        "batch_size": 8,
        "d2h": "synchronous",
        "numeric": "FP32",
        "execution": "eager",
    }
    if config.get("selected_recipe") != fixed_recipe:
        errors.append("production selected recipe is not the fixed smoke recipe")
    row_counts = {1: 121199, 2: 119898, 3: 118901, 4: 118513, 5: 120887}
    expected_binary_bytes = 2 * (64 + row_counts[run_number] * 65536)
    if config.get("expected_binary_output_bytes") != expected_binary_bytes:
        errors.append("production binary output size is not the frozen derivation")
    if config.get("saved_output_limit_bytes") != 20_000_000_000:
        errors.append("production saved-output cap is not Kaggle's 20 GB cap")
    if expected_binary_bytes + 10_000_000 > 20_000_000_000:
        errors.append("production output plus metadata reserve exceeds Kaggle cap")
else:
    expected_id = ""
    errors.append("inference config mode/run combination is invalid")

required_metadata = {
    "id": expected_id,
    "title": (
        "eqvae UBC-OCEAN latent pilot"
        if mode == "pilot"
        else f"eqvae UBC-OCEAN latent run-{run_number:02d}"
        if isinstance(run_number, int)
        else ""
    ),
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": dataset_sources,
    "competition_sources": ["UBC-OCEAN"],
    "kernel_sources": [],
    "model_sources": [],
}
if metadata != required_metadata:
    errors.append("kernel metadata or source order differs from the locked contract")
if "KAGGLE_UBC_OCEAN_LATENT_INFERENCE_READY = True" not in (
    kernel_dir / "run.py"
).read_text(encoding="utf-8"):
    errors.append("generated wrapper is missing the latent inference marker")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PYLATENT
  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --template "$latent_inference_template" \
    --ready-marker "$latent_ready_marker" \
    --verify-only \
    --allow-dirty
}

validate_latent_finalizer_dir() {
  local kernel_dir="$1"
  python3 - \
    "$kernel_dir" \
    "$latent_input_bundle_dir/spec0021_input_contract.json" \
    "$latent_input_receipt" \
    "docs/specs/0021-dual-model-wsi-latent-inference.md" <<'PYLATENTFINALIZER'
import hashlib
import json
import sys
from pathlib import Path

kernel_dir = Path(sys.argv[1])
input_contract = Path(sys.argv[2])
input_receipt_path = Path(sys.argv[3])
spec_path = Path(sys.argv[4])
config_path = kernel_dir / "spec0021_inference_config.json"


def read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected_files = {
    "kernel-metadata.json",
    "run.py",
    "spec0021_inference_config.json",
}
observed_files = {
    path.relative_to(kernel_dir).as_posix()
    for path in kernel_dir.rglob("*")
    if path.is_file()
}
errors: list[str] = []
if observed_files != expected_files:
    errors.append("finalizer upload directory files differ")
config = read_object(config_path)
canonical = json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n"
if config_path.read_text(encoding="utf-8") != canonical:
    errors.append("finalizer config is not canonical compact JSON")
expected_config_fields = {
    "schema_version",
    "mode",
    "spec_sha256",
    "input_contract_sha256",
    "input_dataset_receipt",
    "kernel_sources",
    "production_configs",
}
if set(config) != expected_config_fields:
    errors.append("finalizer config fields differ from the locked schema")
if config.get("schema_version") != "spec0021.finalizer_config.v1":
    errors.append("finalizer config schema mismatch")
if config.get("mode") != "finalizer":
    errors.append("finalizer config mode mismatch")
if config.get("spec_sha256") != sha256(spec_path):
    errors.append("finalizer Spec 0021 hash is stale")
if config.get("input_contract_sha256") != sha256(input_contract):
    errors.append("finalizer input-contract hash is stale")
receipt = read_object(input_receipt_path)
if config.get("input_dataset_receipt") != receipt:
    errors.append("finalizer embedded input receipt differs from local authority")
receipt_version = receipt.get("dataset_version")
if (
    receipt.get("schema_version") != "spec0021.input_dataset_receipt.v1"
    or receipt.get("dataset_reference")
    != "maximusshtefan/eqvae-ubc-ocean-latent-inputs"
    or isinstance(receipt_version, bool)
    or not isinstance(receipt_version, int)
    or receipt_version < 1
):
    errors.append("finalizer input receipt identity differs")
kernel_sources = [
    f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run:02d}"
    for run in range(1, 6)
]
production_fields = {
    "schema_version",
    "mode",
    "run_number",
    "spec_sha256",
    "input_dataset_receipt",
    "input_contract_sha256",
    "work_manifest_sha256",
    "normal_checkpoint_sha256",
    "so2_checkpoint_sha256",
    "pilot_authority_sha256",
    "selected_recipe",
    "expected_binary_output_bytes",
    "saved_output_limit_bytes",
    "resume_dataset_receipt",
}
work_hashes = {
    1: "76cc5f9b86b75b9e46250c80e7b5c98d2f0451a12b38665ae45b055767b7a456",
    2: "11d21482c9d3e083bc5138973c6b09f6b659e7d753853c0290382320e78e6200",
    3: "9414f638abc24963e821de8bbedccaf294a14a7ea768a01687d5b105e634402b",
    4: "e7c5d8d08996e3bac440b5779547b2bbeb4443e7481dcb74998a87aa3054697f",
    5: "5485c44ababeaba9a4e2cc75a1a6b2927d89f10ed83a121dfcb81a68cfc23d6a",
}
normal_checkpoint = "f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075"
so2_checkpoint = "041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7"
row_counts = {1: 121199, 2: 119898, 3: 118901, 4: 118513, 5: 120887}
if config.get("kernel_sources") != kernel_sources:
    errors.append("finalizer kernel-source order differs")
raw_configs = config.get("production_configs")
expected_config_names = {f"run_{run:02d}" for run in range(1, 6)}
if not isinstance(raw_configs, dict) or set(raw_configs) != expected_config_names:
    errors.append("finalizer must bind exactly five production configs")
else:
    shared_bindings = None
    for run in range(1, 6):
        run_name = f"run_{run:02d}"
        record = raw_configs[run_name]
        if not isinstance(record, dict) or set(record) != {"config", "sha256"}:
            errors.append(f"finalizer {run_name} config record differs")
            continue
        production = record["config"]
        if not isinstance(production, dict):
            errors.append(f"finalizer {run_name} config must be an object")
            continue
        encoded = (
            json.dumps(production, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != record["sha256"]:
            errors.append(f"finalizer {run_name} config SHA-256 differs")
        expected = {
            "schema_version": "spec0021.inference_config.v2",
            "mode": "production",
            "run_number": run,
            "spec_sha256": sha256(spec_path),
            "input_contract_sha256": sha256(input_contract),
            "input_dataset_receipt": receipt,
            "work_manifest_sha256": work_hashes[run],
            "normal_checkpoint_sha256": normal_checkpoint,
            "so2_checkpoint_sha256": so2_checkpoint,
        }
        if set(production) != production_fields or any(
            production.get(name) != value for name, value in expected.items()
        ):
            errors.append(f"finalizer {run_name} production binding differs")
        resume = production.get("resume_dataset_receipt")
        if resume is not None:
            expected_resume_reference = (
                f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run:02d}-resume"
            )
            if not isinstance(resume, dict):
                errors.append(f"finalizer {run_name} resume receipt is invalid")
            else:
                expected_resume = {
                    "schema_version": "spec0021.resume_dataset_receipt.v1",
                    "dataset_reference": expected_resume_reference,
                    "run_number": run,
                    "input_bundle_sha256": sha256(input_contract),
                    "work_manifest_sha256": work_hashes[run],
                }
                if any(
                    resume.get(name) != value
                    for name, value in expected_resume.items()
                ):
                    errors.append(
                        f"finalizer {run_name} resume receipt binding differs"
                    )
                version = resume.get("dataset_version")
                if (
                    isinstance(version, bool)
                    or not isinstance(version, int)
                    or version < 1
                ):
                    errors.append(
                        f"finalizer {run_name} resume receipt version is invalid"
                    )
                for name in (
                    "provenance_sha256",
                    "run_config_sha256",
                    "remote_listing_sha256",
                ):
                    value = resume.get(name)
                    if (
                        not isinstance(value, str)
                        or len(value) != 64
                        or any(char not in "0123456789abcdef" for char in value)
                    ):
                        errors.append(
                            f"finalizer {run_name} resume receipt {name} is invalid"
                        )
                remote_files = resume.get("remote_files")
                if not isinstance(remote_files, list) or not remote_files:
                    errors.append(
                        f"finalizer {run_name} resume receipt files are invalid"
                    )
        for name in ("pilot_authority_sha256",):
            value = production.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                errors.append(f"finalizer {run_name} {name} is invalid")
        recipe = production.get("selected_recipe")
        fixed_recipe = {
            "batch_size": 8,
            "d2h": "synchronous",
            "numeric": "FP32",
            "execution": "eager",
        }
        if recipe != fixed_recipe:
            errors.append(f"finalizer {run_name} selected recipe is invalid")
        expected_binary_bytes = 2 * (64 + row_counts[run] * 65536)
        if production.get("expected_binary_output_bytes") != expected_binary_bytes:
            errors.append(f"finalizer {run_name} binary output size is invalid")
        if production.get("saved_output_limit_bytes") != 20_000_000_000:
            errors.append(f"finalizer {run_name} saved-output cap is invalid")
        if expected_binary_bytes + 10_000_000 > 20_000_000_000:
            errors.append(f"finalizer {run_name} output exceeds Kaggle cap")
        observed_shared = {
            name: production.get(name)
            for name in (
                "input_dataset_receipt",
                "normal_checkpoint_sha256",
                "so2_checkpoint_sha256",
                "pilot_authority_sha256",
                "selected_recipe",
                "saved_output_limit_bytes",
            )
        }
        if shared_bindings is None:
            shared_bindings = observed_shared
        elif observed_shared != shared_bindings:
            errors.append("finalizer production shared bindings differ")
metadata = read_object(kernel_dir / "kernel-metadata.json")
expected_metadata = {
    "id": "maximusshtefan/eqvae-ubc-ocean-latent-finalizer",
    "title": "eqvae UBC-OCEAN latent finalizer",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_internet": "true",
    "dataset_sources": ["maximusshtefan/eqvae-ubc-ocean-latent-inputs"],
    "competition_sources": [],
    "kernel_sources": kernel_sources,
    "model_sources": [],
}
if metadata != expected_metadata:
    errors.append("finalizer metadata differs from the locked CPU/source contract")
if "KAGGLE_UBC_OCEAN_LATENT_INFERENCE_READY = True" not in (
    kernel_dir / "run.py"
).read_text(encoding="utf-8"):
    errors.append("finalizer wrapper is missing the latent ready marker")
if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PYLATENTFINALIZER
  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --template "$latent_finalizer_template" \
    --ready-marker "$latent_ready_marker" \
    --verify-only \
    --allow-dirty
}

preflight_latent_inference() {
  local mode="${1:-}"
  case "$mode" in
  pilot)
    validate_latent_kernel_dir "$latent_inference_kernel_root/pilot" allow-null
    ;;
  production-all)
    local run_number
    for run_number in 01 02 03 04 05; do
      validate_latent_kernel_dir \
        "$latent_inference_kernel_root/run_$run_number" fresh
    done
    ;;
  run-0[1-5])
    local run_number="${mode#run-}"
    validate_latent_kernel_dir \
      "$latent_inference_kernel_root/run_$run_number" resume
    ;;
  finalizer)
    validate_latent_finalizer_dir "$latent_inference_kernel_root/finalizer"
    ;;
  *)
    echo "error: preflight-latent-inference requires pilot, production-all, finalizer, or run-XX" >&2
    exit 1
    ;;
  esac
  echo "ok: Spec 0021 latent inference $mode preflight"
}

build_cancer_topup() {
  require_build_python
  "$build_python" -m eqvae.cli.build_ubc_cancer_topup_kernel build \
    --repo-root "$PWD" \
    --plan-root "$cancer_topup_plan_root" \
    --receipt "$cancer_topup_receipt" \
    --output-root "$cancer_topup_kernel_dir"
  preflight_cancer_topup
}

preflight_cancer_topup() {
  require_build_python
  "$build_python" -m eqvae.cli.build_ubc_cancer_topup_kernel validate \
    --repo-root "$PWD" \
    --plan-root "$cancer_topup_plan_root" \
    --receipt "$cancer_topup_receipt" \
    --output-root "$cancer_topup_kernel_dir"
  "$build_python" -m pytest -q tests/test_spec0022_cancer_topup.py
  echo "ok: Spec 0022 one-off cancer top-up preflight"
}

guard_cancer_topup_push_ready() {
  local kernel_dir="$1"
  if [[ "$kernel_dir" != "$cancer_topup_kernel_dir" ]]; then
    echo "error: Spec 0022 push must use $cancer_topup_kernel_dir" >&2
    exit 1
  fi
  if [[ "${KAGGLE_CANCER_TOPUP_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_CANCER_TOPUP_CONFIRMED=1 after explicit one-off authorization" >&2
    exit 1
  fi
  preflight_cancer_topup
}

preflight_mil_capacity_probe() {
  require_build_python
  "$build_python" -m eqvae.cli.build_ubc_mil_capacity_probe validate \
    --repo-root "$PWD" \
    --manifest-root runs/local/ubc_ocean_supervised_manifests \
    --output-root "$mil_capacity_probe_kernel_dir"
  "$build_python" -m pytest -q tests/test_spec0023_mil_capacity_probe.py
  echo "ok: Spec 0023 one-off largest-WSI capacity preflight"
}

guard_mil_capacity_probe_push_ready() {
  local kernel_dir="$1"
  if [[ "$kernel_dir" != "$mil_capacity_probe_kernel_dir" ]]; then
    echo "error: Spec 0023 capacity push must use $mil_capacity_probe_kernel_dir" >&2
    exit 1
  fi
  if [[ "${KAGGLE_MIL_CAPACITY_PROBE_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_MIL_CAPACITY_PROBE_CONFIRMED=1 after explicit authorization" >&2
    exit 1
  fi
  preflight_mil_capacity_probe
}

build_tissue_fastpath_probe() {
  local actor="${1:-}"
  if [[ -z "$actor" ]]; then
    actor="$(kaggle_authenticated_username)"
  fi
  require_build_python
  PYTHONPATH=src "$build_python" scripts/build_tissue_fastpath_probe.py \
    build --actor "$actor"
}

validate_tissue_fastpath_probe() {
  local actor="${1:-}"
  local sealed_source_snapshot="${2:-}"
  require_build_python
  local args=(scripts/build_tissue_fastpath_probe.py validate)
  [[ -n "$actor" ]] && args+=(--actor "$actor")
  [[ "$sealed_source_snapshot" == "sealed" ]] && args+=(--sealed-source-snapshot)
  PYTHONPATH=src "$build_python" "${args[@]}" >/dev/null
}

preflight_tissue_fastpath_probe() {
  local actor="${1:-}"
  validate_tissue_fastpath_probe "$actor" sealed
  validate_kernel_dir "$tissue_fastpath_probe_kernel_dir"
  "$build_python" -m pytest -q tests/test_spec0037_tissue_fastpath_probe.py
  echo "ok: Spec 0037 tissue fast-path probe preflight"
}

publish_tissue_fastpath_probe_inputs() (
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_TISSUE_FASTPATH_PROBE_CONFIRMED:-}" != "1" ]]; then
    echo "error: explicit Spec 0037 Kaggle/dataset confirmations are required" >&2
    exit 1
  fi
  if [[ -e "$tissue_fastpath_probe_input_receipt" ]]; then
    echo "error: immutable Spec 0037 input receipt already exists" >&2
    exit 1
  fi
  local actor create_output
  actor="$(kaggle_authenticated_username)"
  validate_tissue_fastpath_probe "$actor"
  require_kaggle_cli
  if ! create_output="$(
    kaggle_api datasets create -p "$tissue_fastpath_probe_root/upload" 2>&1
  )"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  if [[ "$create_output" == *"Dataset creation error"* ]]; then
    echo "error: Kaggle reported Spec 0037 dataset creation failure" >&2
    exit 1
  fi
)

verify_tissue_fastpath_probe_inputs() (
  require_remote_confirmed
  local actor dataset_reference stage_parent download_dir metadata_dir status_path
  local dataset_version
  actor="$(kaggle_authenticated_username)"
  dataset_reference="$actor/$tissue_fastpath_probe_dataset_slug"
  validate_tissue_fastpath_probe "$actor" sealed
  if [[ -e "$tissue_fastpath_probe_input_receipt" ]]; then
    echo "error: immutable Spec 0037 input receipt already exists" >&2
    exit 1
  fi
  require_kaggle_cli
  stage_parent="$(mktemp -d "$TMPDIR/spec0037_input_verify.XXXXXX")"
  trap 'rm -rf -- "$stage_parent"' EXIT
  download_dir="$stage_parent/download"
  metadata_dir="$stage_parent/metadata"
  status_path="$stage_parent/status.json"
  mkdir -p "$download_dir" "$metadata_dir" "$tissue_fastpath_probe_authority_root"
  kaggle_api datasets status "$dataset_reference" \
    --format 'json(status,current_version_number)' >"$status_path"
  dataset_version="$("$build_python" - "$status_path" <<'PYSPEC0037VERSION'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("status") != "ready" or status.get("current_version_number") != 1:
    raise SystemExit("remote Spec 0037 input must be ready immutable version 1")
print(1)
PYSPEC0037VERSION
)"
  kaggle_api datasets metadata "$dataset_reference" -p "$metadata_dir"
  kaggle_api datasets download "$dataset_reference/$dataset_version" \
    -p "$download_dir" --unzip -o -q
  "$build_python" - \
    "$tissue_fastpath_probe_root" "$download_dir" \
    "$metadata_dir/dataset-metadata.json" "$dataset_reference" \
    "$tissue_fastpath_probe_input_receipt" <<'PYSPEC0037RECEIPT'
import hashlib
import json
import os
import sys
from pathlib import Path

root, downloaded_root, metadata_path, reference, receipt_path = map(Path, sys.argv[1:])
reference = str(reference)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


bundle = root / "bundle"
expected = {
    path.relative_to(bundle).as_posix(): path
    for path in bundle.rglob("*")
    if path.is_file() and path.name != "dataset-metadata.json"
}
downloaded = {
    path.relative_to(downloaded_root).as_posix(): path
    for path in downloaded_root.rglob("*")
    if path.is_file() and path.name != "dataset-metadata.json"
}
if set(downloaded) != set(expected):
    raise SystemExit("downloaded Spec 0037 input allow-list differs")
remote_files = []
for name, local in sorted(expected.items()):
    remote = downloaded[name]
    if local.stat().st_size != remote.stat().st_size or sha256(local) != sha256(remote):
        raise SystemExit(f"downloaded Spec 0037 input differs: {name}")
    remote_files.append({"logical_name": name, "bytes": local.stat().st_size, "sha256": sha256(local)})
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
info = metadata.get("info", {})
owner, slug = reference.split("/", maxsplit=1)
if not isinstance(info, dict) or info.get("ownerUser") != owner or info.get("datasetSlug") != slug or info.get("isPrivate") is not True:
    raise SystemExit("remote Spec 0037 input identity/privacy differs")
contract = bundle / "tissue_fastpath_probe_input.json"
record = {
    "schema_version": "spec0037.input_dataset_receipt.v1",
    "dataset_reference": reference,
    "dataset_version": 1,
    "visibility": "private",
    "status": "verified",
    "input_contract_sha256": sha256(contract),
    "remote_files": remote_files,
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(receipt_path)
directory = os.open(receipt_path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0037RECEIPT
  echo "ok: wrote verified immutable Spec 0037 input receipt $tissue_fastpath_probe_input_receipt"
)

build_tissue_training() {
  local actor="${1:-}"
  if [[ -z "$actor" ]]; then
    actor="$(kaggle_authenticated_username)"
  fi
  require_build_python
  PYTHONPATH=src "$build_python" scripts/build_tissue_training.py build --actor "$actor"
}

validate_tissue_training() {
  local actor="${1:-}"
  local sealed_source_snapshot="${2:-}"
  require_build_python
  local args=(scripts/build_tissue_training.py validate)
  [[ -n "$actor" ]] && args+=(--actor "$actor")
  [[ "$sealed_source_snapshot" == "sealed" ]] && args+=(--sealed-source-snapshot)
  PYTHONPATH=src "$build_python" "${args[@]}" >/dev/null
}

preflight_tissue_training() {
  local actor="${1:-}"
  validate_tissue_training "$actor" sealed
  validate_kernel_dir "$tissue_training_kernel_dir"
  "$build_python" -m pytest -q tests/test_spec0039_tissue_training.py
  echo "ok: Spec 0039 tissue label-efficiency training preflight"
}

build_tissue_training_retry() {
  local actor="${1:?authenticated Kaggle actor required}"
  local input_bundle="${2:?frozen input bundle required}"
  require_build_python
  PYTHONPATH=src "$build_python" scripts/build_tissue_training.py build-retry \
    --actor "$actor" --input-bundle "$input_bundle" \
    --output-root "$tissue_training_retry_root"
}

validate_tissue_training_retry() {
  local actor="${1:?authenticated Kaggle actor required}"
  require_build_python
  PYTHONPATH=src "$build_python" scripts/build_tissue_training.py validate-retry \
    --actor "$actor" --output-root "$tissue_training_retry_root" >/dev/null
}

preflight_tissue_training_retry() {
  local actor="${1:-}"
  if [[ -z "$actor" ]]; then
    actor="$(kaggle_authenticated_username)"
  fi
  validate_tissue_training_retry "$actor"
  validate_kernel_dir "$tissue_training_retry_kernel_dir"
  "$build_python" -m pytest -q tests/test_spec0039_tissue_training.py
  echo "ok: Spec 0039 tissue label-efficiency training retry-v2 preflight"
}

build_tissue_training_retry_v3() {
  local actor="${1:?authenticated Kaggle actor required}"
  local input_bundle="${2:?frozen input bundle required}"
  require_build_python
  PYTHONPATH=src "$build_python" scripts/build_tissue_training.py build-retry \
    --actor "$actor" --input-bundle "$input_bundle" \
    --output-root "$tissue_training_retry_v3_root"
}

validate_tissue_training_retry_v3() {
  local actor="${1:?authenticated Kaggle actor required}"
  require_build_python
  PYTHONPATH=src "$build_python" scripts/build_tissue_training.py validate-retry \
    --actor "$actor" --output-root "$tissue_training_retry_v3_root" >/dev/null
}

preflight_tissue_training_retry_v3() {
  local actor="${1:-}"
  if [[ -z "$actor" ]]; then
    actor="$(kaggle_authenticated_username)"
  fi
  validate_tissue_training_retry_v3 "$actor"
  validate_kernel_dir "$tissue_training_retry_v3_kernel_dir"
  "$build_python" -m pytest -q tests/test_spec0039_tissue_training.py
  echo "ok: Spec 0039 tissue label-efficiency training retry-v3 preflight"
}

publish_tissue_training_inputs() (
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_TISSUE_TRAINING_CONFIRMED:-}" != "1" ]]; then
    echo "error: explicit Spec 0039 Kaggle/dataset confirmations are required" >&2
    exit 1
  fi
  if [[ -e "$tissue_training_input_receipt" ]]; then
    echo "error: immutable Spec 0039 input receipt already exists" >&2
    exit 1
  fi
  local actor create_output
  actor="$(kaggle_authenticated_username)"
  validate_tissue_training "$actor"
  require_kaggle_cli
  if ! create_output="$(
    kaggle_api datasets create -p "$tissue_training_root/upload" 2>&1
  )"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  if [[ "$create_output" == *"Dataset creation error"* ]]; then
    echo "error: Kaggle reported Spec 0039 dataset creation failure" >&2
    exit 1
  fi
)

verify_tissue_training_inputs() (
  require_remote_confirmed
  local actor dataset_reference stage_parent download_dir metadata_dir status_path
  local dataset_version
  actor="$(kaggle_authenticated_username)"
  dataset_reference="$actor/$tissue_training_dataset_slug"
  validate_tissue_training "$actor" sealed
  if [[ -e "$tissue_training_input_receipt" ]]; then
    echo "error: immutable Spec 0039 input receipt already exists" >&2
    exit 1
  fi
  require_kaggle_cli
  stage_parent="$(mktemp -d "$TMPDIR/spec0039_input_verify.XXXXXX")"
  trap 'rm -rf -- "$stage_parent"' EXIT
  download_dir="$stage_parent/download"
  metadata_dir="$stage_parent/metadata"
  status_path="$stage_parent/status.json"
  mkdir -p "$download_dir" "$metadata_dir" "$tissue_training_authority_root"
  kaggle_api datasets status "$dataset_reference" \
    --format 'json(status,current_version_number)' >"$status_path"
  dataset_version="$("$build_python" - "$status_path" <<'PYSPEC0039VERSION'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("status") != "ready" or status.get("current_version_number") != 1:
    raise SystemExit("remote Spec 0039 input must be ready immutable version 1")
print(1)
PYSPEC0039VERSION
)"
  kaggle_api datasets metadata "$dataset_reference" -p "$metadata_dir"
  kaggle_api datasets download "$dataset_reference/$dataset_version" \
    -p "$download_dir" --unzip -o -q
  "$build_python" - \
    "$tissue_training_root" "$download_dir" \
    "$metadata_dir/dataset-metadata.json" "$dataset_reference" \
    "$tissue_training_input_receipt" <<'PYSPEC0039RECEIPT'
import hashlib
import json
import os
import sys
from pathlib import Path

root, downloaded_root, metadata_path, reference, receipt_path = map(Path, sys.argv[1:])
reference = str(reference)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


bundle = root / "bundle"
expected = {
    path.relative_to(bundle).as_posix(): path
    for path in bundle.rglob("*")
    if path.is_file() and path.name != "dataset-metadata.json"
}
downloaded = {
    path.relative_to(downloaded_root).as_posix(): path
    for path in downloaded_root.rglob("*")
    if path.is_file() and path.name != "dataset-metadata.json"
}
if set(downloaded) != set(expected):
    raise SystemExit("downloaded Spec 0039 input allow-list differs")
remote_files = []
for name, local in sorted(expected.items()):
    remote = downloaded[name]
    if local.stat().st_size != remote.stat().st_size or sha256(local) != sha256(remote):
        raise SystemExit(f"downloaded Spec 0039 input differs: {name}")
    remote_files.append({"logical_name": name, "bytes": local.stat().st_size, "sha256": sha256(local)})
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
info = metadata.get("info", {})
owner, slug = reference.split("/", maxsplit=1)
if not isinstance(info, dict) or info.get("ownerUser") != owner or info.get("datasetSlug") != slug or info.get("isPrivate") is not True:
    raise SystemExit("remote Spec 0039 input identity/privacy differs")
contract = bundle / "tissue_training_input.json"
record = {
    "schema_version": "spec0039.input_dataset_receipt.v1",
    "dataset_reference": reference,
    "dataset_version": 1,
    "visibility": "private",
    "status": "verified",
    "input_contract_sha256": sha256(contract),
    "remote_files": remote_files,
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(receipt_path)
directory = os.open(receipt_path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0039RECEIPT
  echo "ok: wrote verified immutable Spec 0039 input receipt $tissue_training_input_receipt"
)

output_tissue_training() {
  local receipt="${1:?launch receipt required}"
  local output_dir="${2:?output directory required}"
  local kernel_reference
  kernel_reference="$(kernel_reference_from_launch_receipt "$receipt")"
  if [[ ! "$kernel_reference" =~ ^[^/]+/eqvae-tissue-label-efficiency-training/[0-9]+$ ]]; then
    echo "error: launch receipt is not for Spec 0039 tissue training" >&2
    exit 1
  fi
  require_remote_confirmed
  require_kaggle_cli
  if [[ -e "$output_dir" ]]; then
    echo "error: output-tissue-training requires a new output directory" >&2
    exit 1
  fi
  mkdir -p "$output_dir"
  kaggle_api kernels output "$kernel_reference" -p "$output_dir"
  record_kaggle_download \
    kernel "$kernel_reference" "$output_dir" kaggle_output_receipt.json
}

build_mil_test() {
  local actor="${1:-}"
  [[ -n "$actor" ]] || actor="$(kaggle_authenticated_username)"
  require_build_python
  "$build_python" scripts/build_ubc_mil_test_evaluation.py build --actor "$actor"
}

validate_mil_test() {
  local actor="${1:-}"
  require_build_python
  local args=(scripts/build_ubc_mil_test_evaluation.py validate)
  [[ -z "$actor" ]] || args+=(--actor "$actor")
  "$build_python" "${args[@]}"
}

publish_mil_test_inputs() (
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_MIL_TEST_EVALUATION_CONFIRMED:-}" != "1" ]]; then
    echo "error: Spec 0041 dataset publication confirmations are required" >&2
    exit 1
  fi
  [[ ! -e "$mil_test_input_receipt" ]] || {
    echo "error: immutable Spec 0041 input receipt already exists" >&2
    exit 1
  }
  local actor create_output
  actor="$(kaggle_authenticated_username)"
  validate_mil_test "$actor" >/dev/null
  require_kaggle_cli
  if ! create_output="$(
    kaggle_api datasets create -p "$mil_test_root/upload" 2>&1
  )"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  [[ "$create_output" != *"Dataset creation error"* ]] || exit 1
)

status_mil_test_inputs() {
  require_remote_confirmed
  local actor
  actor="$(kaggle_authenticated_username)"
  kaggle_api datasets status "$actor/$mil_test_dataset_slug" \
    --format 'json(status,current_version_number)'
}

verify_mil_test_inputs() (
  require_remote_confirmed
  local actor reference stage download metadata status version
  actor="$(kaggle_authenticated_username)"
  reference="$actor/$mil_test_dataset_slug"
  validate_mil_test "$actor" >/dev/null
  [[ ! -e "$mil_test_input_receipt" ]] || {
    echo "error: immutable Spec 0041 input receipt already exists" >&2
    exit 1
  }
  stage="$(mktemp -d "$TMPDIR/spec0041_verify.XXXXXX")"
  trap 'rm -rf -- "$stage"' EXIT
  download="$stage/download"
  metadata="$stage/metadata"
  status="$stage/status.json"
  mkdir -p "$download" "$metadata"
  kaggle_api datasets status "$reference" \
    --format 'json(status,current_version_number)' >"$status"
  version="$(python3 - "$status" <<'PYSPEC0041VERSION'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("status") != "ready" or value.get("current_version_number") != 1:
    raise SystemExit("Spec 0041 dataset must be ready immutable version 1")
print(1)
PYSPEC0041VERSION
)"
  kaggle_api datasets metadata "$reference" -p "$metadata"
  kaggle_api datasets download "$reference/$version" -p "$download" --unzip -o -q
  "$build_python" - "$mil_test_root/bundle" "$download" \
    "$metadata/dataset-metadata.json" "$reference" "$mil_test_input_receipt" <<'PYSPEC0041RECEIPT'
import hashlib
import json
import os
import sys
from pathlib import Path

local, remote, metadata_path, reference, receipt = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
)

def record(root):
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in root.rglob("*")
        if path.is_file() and path.name != "dataset-metadata.json"
    }

local_files = record(local)
remote_files = record(remote)
if local_files != remote_files:
    raise SystemExit("downloaded Spec 0041 dataset bytes differ")
metadata = json.loads(metadata_path.read_text(encoding="utf-8")).get("info", {})
owner, slug = reference.split("/", 1)
if metadata.get("ownerUser") != owner or metadata.get("datasetSlug") != slug or metadata.get("isPrivate") is not True:
    raise SystemExit("remote Spec 0041 identity/privacy differs")
contract = local / "mil_test_inference_input.json"
value = {
    "schema_version": "spec0041.input_dataset_receipt.v1",
    "dataset_reference": reference,
    "dataset_version": 1,
    "visibility": "private",
    "status": "verified",
    "input_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "remote_files": remote_files,
}
receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
directory = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0041RECEIPT
  echo "ok: wrote verified immutable Spec 0041 input receipt $mil_test_input_receipt"
)

output_mil_test() {
  local receipt="${1:?launch receipt required}"
  local output_dir="${2:?output directory required}"
  local reference
  reference="$(kernel_reference_from_launch_receipt "$receipt")"
  local receipt_sha256
  receipt_sha256="$(sha256sum "$receipt" | awk '{print $1}')"
  [[ "$reference" == "$mil_test_accepted_reference" \
    && "$receipt_sha256" == "$mil_test_launch_receipt_sha256" ]] || {
    echo "error: launch receipt is not the exact amended Spec 0041 launch" >&2
    exit 1
  }
  require_remote_confirmed
  [[ ! -e "$output_dir" ]] || {
    echo "error: output-mil-test requires a new output directory" >&2
    exit 1
  }
  mkdir -p "$output_dir"
  kaggle_api kernels output "$reference" -p "$output_dir"
  record_kaggle_download kernel "$reference" "$output_dir" kaggle_output_receipt.json
}

score_mil_test() {
  require_build_python
  "$build_python" scripts/build_ubc_mil_test_evaluation.py score \
    --remote-output-root "${1:?remote output required}" \
    --launch-receipt "${2:?launch receipt required}" \
    --output-root "${3:?scored output required}"
}

build_tissue_test() {
  local actor="${1:-}"
  [[ -n "$actor" ]] || actor="$(kaggle_authenticated_username)"
  require_build_python
  "$build_python" scripts/build_tissue_test_evaluation.py build --actor "$actor"
}

validate_tissue_test() {
  local actor="${1:-}"
  require_build_python
  local args=(scripts/build_tissue_test_evaluation.py validate)
  [[ -z "$actor" ]] || args+=(--actor "$actor")
  "$build_python" "${args[@]}"
}

publish_tissue_test_inputs() (
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_TISSUE_TEST_EVALUATION_CONFIRMED:-}" != "1" ]]; then
    echo "error: Spec 0043 dataset publication confirmations are required" >&2
    exit 1
  fi
  [[ ! -e "$tissue_test_input_receipt" ]] || {
    echo "error: immutable Spec 0043 input receipt already exists" >&2
    exit 1
  }
  local actor create_output
  actor="$(kaggle_authenticated_username)"
  validate_tissue_test "$actor" >/dev/null
  require_kaggle_cli
  if ! create_output="$(
    kaggle_api datasets create -p "$tissue_test_root/upload" 2>&1
  )"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  [[ "$create_output" != *"Dataset creation error"* ]] || exit 1
)

status_tissue_test_inputs() {
  require_remote_confirmed
  local actor
  actor="$(kaggle_authenticated_username)"
  kaggle_api datasets status "$actor/$tissue_test_dataset_slug" \
    --format 'json(status,current_version_number)'
}

verify_tissue_test_inputs() (
  require_remote_confirmed
  local actor reference stage download metadata status version
  actor="$(kaggle_authenticated_username)"
  reference="$actor/$tissue_test_dataset_slug"
  validate_tissue_test "$actor" >/dev/null
  [[ ! -e "$tissue_test_input_receipt" ]] || {
    echo "error: immutable Spec 0043 input receipt already exists" >&2
    exit 1
  }
  stage="$(mktemp -d "$TMPDIR/spec0043_verify.XXXXXX")"
  trap 'rm -rf -- "$stage"' EXIT
  download="$stage/download"
  metadata="$stage/metadata"
  status="$stage/status.json"
  mkdir -p "$download" "$metadata"
  kaggle_api datasets status "$reference" \
    --format 'json(status,current_version_number)' >"$status"
  version="$(python3 - "$status" <<'PYSPEC0043VERSION'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("status") != "ready" or value.get("current_version_number") != 1:
    raise SystemExit("Spec 0043 dataset must be ready immutable version 1")
print(1)
PYSPEC0043VERSION
)"
  kaggle_api datasets metadata "$reference" -p "$metadata"
  kaggle_api datasets download "$reference/$version" -p "$download" --unzip -o -q
  "$build_python" - "$tissue_test_root/bundle" "$download" \
    "$metadata/dataset-metadata.json" "$reference" "$tissue_test_input_receipt" <<'PYSPEC0043RECEIPT'
import hashlib
import json
import os
import sys
from pathlib import Path

local, remote, metadata_path, reference, receipt = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
)

def record(root):
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in root.rglob("*")
        if path.is_file() and path.name != "dataset-metadata.json"
    }

local_files = record(local)
remote_files = record(remote)
if local_files != remote_files:
    raise SystemExit("downloaded Spec 0043 dataset bytes differ")
metadata = json.loads(metadata_path.read_text(encoding="utf-8")).get("info", {})
owner, slug = reference.split("/", 1)
if (
    metadata.get("ownerUser") != owner
    or metadata.get("datasetSlug") != slug
    or metadata.get("isPrivate") is not True
):
    raise SystemExit("remote Spec 0043 identity/privacy differs")
contract = local / "tissue_test_inference_input.json"
value = {
    "schema_version": "spec0043.input_dataset_receipt.v1",
    "dataset_reference": reference,
    "dataset_version": 1,
    "visibility": "private",
    "status": "verified",
    "input_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "remote_files": remote_files,
}
receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
directory = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0043RECEIPT
  echo "ok: wrote verified immutable Spec 0043 input receipt $tissue_test_input_receipt"
)

push_tissue_test() {
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_TISSUE_TEST_EVALUATION_CONFIRMED:-}" != "1" ]]; then
    echo "error: Spec 0043 kernel launch confirmations are required" >&2
    exit 1
  fi
  [[ -f "$tissue_test_input_receipt" ]] || {
    echo "error: Spec 0043 verified input receipt is required" >&2
    exit 1
  }
  [[ ! -e "$tissue_test_launch_claim" ]] || {
    echo "error: Spec 0043 one-use launch claim already exists" >&2
    exit 1
  }
  local actor
  actor="$(kaggle_authenticated_username)"
  require_build_python
  "$build_python" scripts/build_tissue_test_evaluation.py claim-launch --actor "$actor"
  KAGGLE_TISSUE_TEST_ROUTE_ACTIVE=1 "$0" push "$tissue_test_kernel_dir"
}

output_tissue_test() {
  local receipt="${1:?launch receipt required}"
  local output_dir="${2:?output directory required}"
  local reference
  reference="$(kernel_reference_from_launch_receipt "$receipt")"
  if [[ ! "$reference" =~ ^[^/]+/$tissue_test_kernel_slug/[0-9]+$ ]]; then
    echo "error: launch receipt is not for Spec 0043 tissue test" >&2
    exit 1
  fi
  require_remote_confirmed
  require_kaggle_cli
  [[ ! -e "$output_dir" ]] || {
    echo "error: output-tissue-test requires a new output directory" >&2
    exit 1
  }
  mkdir -p "$output_dir"
  kaggle_api kernels output "$reference" -p "$output_dir"
  record_kaggle_download kernel "$reference" "$output_dir" kaggle_output_receipt.json
}

score_tissue_test() {
  require_build_python
  "$build_python" scripts/build_tissue_test_evaluation.py score \
    --remote-output-root "${1:?remote output required}" \
    --launch-receipt "${2:?launch receipt required}" \
    --output-root "${3:?scored output required}"
}

validate_vae_test() {
  local actor="${1:-}"
  [[ -n "$actor" ]] || actor="$(kaggle_authenticated_username)"
  require_build_python
  "$build_python" scripts/build_vae_test_evaluation.py \
    validate-input --actor "$actor"
  build_kernel_py \
    --kernel-dir "$vae_test_kernel_dir" \
    --ready-marker "SPEC0045_VAE_TEST_RECONSTRUCTION_READY = True" \
    --allow-dirty --verify-only
}

publish_vae_test_inputs() (
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_VAE_TEST_EVALUATION_CONFIRMED:-}" != "1" ]]; then
    echo "error: Spec 0045 dataset publication confirmations are required" >&2
    exit 1
  fi
  [[ ! -e "$vae_test_input_receipt" ]] || {
    echo "error: immutable Spec 0045 input receipt already exists" >&2
    exit 1
  }
  local actor create_output
  actor="$(kaggle_authenticated_username)"
  validate_vae_test "$actor" >/dev/null
  require_kaggle_cli
  if ! create_output="$(
    kaggle_api datasets create -p "$vae_test_input_root" 2>&1
  )"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  [[ "$create_output" != *"Dataset creation error"* ]] || exit 1
)

status_vae_test_inputs() {
  require_remote_confirmed
  local actor
  actor="$(kaggle_authenticated_username)"
  kaggle_api datasets status "$actor/$vae_test_dataset_slug" \
    --format 'json(status,current_version_number)'
}

verify_vae_test_inputs() (
  require_remote_confirmed
  local actor reference stage download metadata status
  actor="$(kaggle_authenticated_username)"
  reference="$actor/$vae_test_dataset_slug"
  validate_vae_test "$actor" >/dev/null
  [[ ! -e "$vae_test_input_receipt" ]] || {
    echo "error: immutable Spec 0045 input receipt already exists" >&2
    exit 1
  }
  stage="$(mktemp -d "$TMPDIR/spec0045_verify.XXXXXX")"
  trap 'rm -rf -- "$stage"' EXIT
  download="$stage/download"
  metadata="$stage/metadata"
  status="$stage/status.json"
  mkdir -p "$download" "$metadata" "$vae_test_authority_root"
  kaggle_api datasets status "$reference" \
    --format 'json(status,current_version_number)' >"$status"
  "$build_python" - "$status" <<'PYSPEC0045VERSION'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("status") != "ready" or value.get("current_version_number") != 1:
    raise SystemExit("Spec 0045 dataset must be ready immutable version 1")
PYSPEC0045VERSION
  kaggle_api datasets metadata "$reference" -p "$metadata"
  kaggle_api datasets download "$reference/1" -p "$download" --unzip -o -q
  "$build_python" - "$vae_test_input_root" "$download" \
    "$metadata/dataset-metadata.json" "$reference" "$vae_test_input_receipt" \
    <<'PYSPEC0045RECEIPT'
import hashlib
import json
import os
import sys
from pathlib import Path

local, remote, metadata_path, reference, receipt = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
)

def record(root):
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in root.rglob("*")
        if path.is_file() and path.name != "dataset-metadata.json"
    }

local_files = record(local)
remote_files = record(remote)
if local_files != remote_files:
    raise SystemExit("downloaded Spec 0045 dataset bytes differ")
metadata = json.loads(metadata_path.read_text(encoding="utf-8")).get("info", {})
owner, slug = reference.split("/", 1)
if (
    metadata.get("ownerUser") != owner
    or metadata.get("datasetSlug") != slug
    or metadata.get("isPrivate") is not True
):
    raise SystemExit("remote Spec 0045 identity/privacy differs")
contract = local / "spec0045_vae_test_input.json"
value = {
    "schema_version": "spec0045.input_dataset_receipt.v1",
    "dataset_reference": reference,
    "dataset_version": 1,
    "visibility": "private",
    "status": "verified",
    "input_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "remote_files": remote_files,
}
with receipt.open("x", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PYSPEC0045RECEIPT
  echo "ok: wrote verified immutable Spec 0045 input receipt $vae_test_input_receipt"
)

push_vae_test() {
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_VAE_TEST_EVALUATION_CONFIRMED:-}" != "1" ]]; then
    echo "error: Spec 0045 kernel launch confirmations are required" >&2
    exit 1
  fi
  [[ -f "$vae_test_input_receipt" ]] || {
    echo "error: Spec 0045 verified input receipt is required" >&2
    exit 1
  }
  [[ ! -e "$vae_test_launch_claim" ]] || {
    echo "error: Spec 0045 one-use launch claim already exists" >&2
    exit 1
  }
  local actor
  actor="$(kaggle_authenticated_username)"
  require_build_python
  "$build_python" scripts/build_vae_test_evaluation.py \
    claim-launch --actor "$actor"
  KAGGLE_VAE_TEST_ROUTE_ACTIVE=1 "$0" push "$vae_test_kernel_dir"
}

output_vae_test() {
  local receipt="${1:?launch receipt required}"
  local output_dir="${2:?output directory required}"
  local reference receipt_sha256
  reference="$(kernel_reference_from_launch_receipt "$receipt")"
  receipt_sha256="$(sha256sum "$receipt" | awk '{print $1}')"
  if [[ "$reference" != "$vae_test_accepted_reference" \
    || "$receipt_sha256" != "$vae_test_launch_receipt_sha256" ]]; then
    echo "error: launch receipt is not the one-shot Spec 0045 VAE test" >&2
    exit 1
  fi
  require_remote_confirmed
  require_kaggle_cli
  [[ ! -e "$output_dir" ]] || {
    echo "error: output-vae-test requires a new output directory" >&2
    exit 1
  }
  mkdir -p "$output_dir"
  kaggle_api kernels output "$reference" -p "$output_dir"
  record_kaggle_download kernel "$reference" "$output_dir" kaggle_output_receipt.json
}

score_vae_test() {
  require_build_python
  "$build_python" scripts/build_vae_test_evaluation.py score \
    --remote-output-root "${1:?remote output required}" \
    --launch-receipt "${2:?launch receipt required}" \
    --input-receipt "$vae_test_input_receipt" \
    --output-root "${3:?scored output required}"
}

resume_score_vae_test() {
  require_build_python
  "$build_python" scripts/build_vae_test_evaluation.py resume-score \
    --remote-output-root runs/kaggle/vae_test_reconstruction_v1 \
    --launch-receipt runs/local/kaggle_launches/maximshtefan/eqvae-frozen-vae-full-test-reconstruction/v0001.json \
    --input-receipt "$vae_test_input_receipt" \
    --output-root runs/local/vae_test_reconstruction_scored_v1
}

build_mil_training() {
  local actor="${1:-}"
  if [[ -z "$actor" ]]; then
    actor="$(kaggle_authenticated_username)"
  fi
  require_build_python
  "$build_python" scripts/build_ubc_mil_training.py build --actor "$actor"
}

validate_mil_training() {
  local actor="${1:-}"
  require_build_python
  local args=(scripts/build_ubc_mil_training.py validate)
  if [[ -n "$actor" ]]; then
    args+=(--actor "$actor")
  fi
  "$build_python" "${args[@]}" >/dev/null
}

build_mil_training_resume() {
  local actor="${1:-}"
  local output_root="${2:-}"
  local launch_receipt="${3:-}"
  if [[ -z "$actor" || -z "$output_root" || -z "$launch_receipt" ]]; then
    echo "error: build-mil-training-resume requires actor, prior-output-root, and prior-launch-receipt" >&2
    return 2
  fi
  require_build_python
  "$build_python" scripts/build_ubc_mil_training.py build-resume \
    --actor "$actor" --output-root "$output_root" \
    --launch-receipt "$launch_receipt"
}

validate_mil_training_resume() {
  local actor="${1:-}"
  local resume_root="${2:-}"
  if [[ -z "$resume_root" ]]; then
    echo "error: validate-mil-training-resume requires actor and resume-root" >&2
    return 2
  fi
  require_build_python
  local args=(scripts/build_ubc_mil_training.py validate-resume \
    --output-root "$resume_root")
  [[ -n "$actor" ]] && args+=(--actor "$actor")
  "$build_python" "${args[@]}"
}

publish_mil_training_resume() (
  local resume_root="${1:?resume package root required}"
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" ]]; then
    echo "error: Kaggle push and dataset-write confirmations are required" >&2
    exit 1
  fi
  local actor create_output
  actor="$(kaggle_authenticated_username)"
  validate_mil_training_resume "$actor" "$resume_root" >/dev/null
  if [[ -e "$resume_root/resume_dataset_receipt.json" ]]; then
    echo "error: immutable Spec 0036 resume receipt already exists" >&2
    exit 1
  fi
  require_kaggle_cli
  if ! create_output="$(
    kaggle_api datasets create -p "$resume_root/upload" 2>&1
  )"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  if [[ "$create_output" == *"Dataset creation error"* ]]; then
    echo "error: Kaggle reported Spec 0036 resume dataset creation failure" >&2
    exit 1
  fi
)

verify_mil_training_resume() (
  local resume_root="${1:?resume package root required}"
  require_remote_confirmed
  local actor dataset_reference stage_parent download_dir metadata_dir
  local status_path dataset_version receipt_path
  actor="$(kaggle_authenticated_username)"
  validate_mil_training_resume "$actor" "$resume_root" >/dev/null
  dataset_reference="$(
    json_field "$resume_root/bundle/mil_training_resume.json" dataset_reference
  )"
  receipt_path="$resume_root/resume_dataset_receipt.json"
  if [[ -e "$receipt_path" ]]; then
    echo "error: immutable Spec 0036 resume receipt already exists" >&2
    exit 1
  fi
  require_kaggle_cli
  stage_parent="$(mktemp -d "$TMPDIR/spec0036_resume_verify.XXXXXX")"
  trap 'rm -rf -- "$stage_parent"' EXIT
  download_dir="$stage_parent/download"
  metadata_dir="$stage_parent/metadata"
  status_path="$stage_parent/status.json"
  mkdir -p "$download_dir" "$metadata_dir"
  kaggle_api datasets status "$dataset_reference" \
    --format 'json(status,current_version_number)' >"$status_path"
  dataset_version="$(python3 - "$status_path" <<'PYSPEC0036RESUMEVERSION'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("status") != "ready" or status.get("current_version_number") != 1:
    raise SystemExit("remote Spec 0036 resume dataset must be ready immutable version 1")
print(1)
PYSPEC0036RESUMEVERSION
)"
  kaggle_api datasets metadata "$dataset_reference" -p "$metadata_dir"
  kaggle_api datasets download "$dataset_reference/$dataset_version" \
    -p "$download_dir" --unzip -o -q
  "$build_python" - \
    "$resume_root" "$download_dir" \
    "$metadata_dir/dataset-metadata.json" "$dataset_reference" \
    "$dataset_version" "$receipt_path" <<'PYSPEC0036RESUMERECEIPT'
import hashlib
import json
import os
import sys
from pathlib import Path

root, downloaded_root, metadata_path, reference, version_text, receipt_path = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
    Path(sys.argv[6]),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


bundle = root / "bundle"
downloaded = {
    path.relative_to(downloaded_root).as_posix(): path
    for path in downloaded_root.rglob("*")
    if path.is_file() and path.name != "dataset-metadata.json"
}
expected = {
    path.relative_to(bundle).as_posix(): path
    for path in bundle.rglob("*")
    if path.is_file() and path.name != "dataset-metadata.json"
}
if set(downloaded) != set(expected):
    raise SystemExit("downloaded Spec 0036 resume allow-list differs")
remote_files = []
for name, local_path in sorted(expected.items()):
    remote_path = downloaded[name]
    if (
        local_path.stat().st_size != remote_path.stat().st_size
        or sha256(local_path) != sha256(remote_path)
    ):
        raise SystemExit(f"downloaded Spec 0036 resume differs: {name}")
    remote_files.append(
        {"logical_name": name, "bytes": local_path.stat().st_size, "sha256": sha256(local_path)}
    )
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
info = metadata.get("info", {})
owner, slug = reference.split("/", maxsplit=1)
if (
    not isinstance(info, dict)
    or info.get("ownerUser") != owner
    or info.get("datasetSlug") != slug
    or info.get("isPrivate") is not True
):
    raise SystemExit("remote Spec 0036 resume identity/privacy differs")
contract_path = bundle / "mil_training_resume.json"
record = {
    "schema_version": "spec0036.resume_dataset_receipt.v1",
    "dataset_reference": reference,
    "dataset_version": int(version_text),
    "visibility": "private",
    "status": "verified",
    "resume_contract_sha256": sha256(contract_path),
    "remote_files": remote_files,
}
temporary = receipt_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(receipt_path)
directory = os.open(receipt_path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0036RESUMERECEIPT
  echo "ok: wrote verified immutable Spec 0036 resume receipt $receipt_path"
)

preflight_mil_training() {
  local actor="${1:-}"
  validate_mil_training "$actor"
  validate_kernel_dir "$mil_training_kernel_dir"
  "$build_python" -m pytest -q \
    tests/test_spec0036_mil_training.py \
    tests/test_spec0036_mil_package.py
  echo "ok: Spec 0036 local-global MIL training preflight"
}

publish_mil_training_inputs() (
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" ]]; then
    echo "error: Kaggle push and dataset-write confirmations are required" >&2
    exit 1
  fi
  if [[ -e "$mil_training_input_receipt" ]]; then
    echo "error: immutable Spec 0036 input receipt already exists" >&2
    exit 1
  fi
  local actor create_output
  actor="$(kaggle_authenticated_username)"
  validate_mil_training "$actor"
  require_kaggle_cli
  if ! create_output="$(
    kaggle_api datasets create -p "$mil_training_root/upload" 2>&1
  )"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  if [[ "$create_output" == *"Dataset creation error"* ]]; then
    echo "error: Kaggle reported Spec 0036 dataset creation failure" >&2
    exit 1
  fi
)

verify_mil_training_inputs() (
  require_remote_confirmed
  local actor dataset_reference stage_parent download_dir metadata_dir
  local status_path dataset_version
  actor="$(kaggle_authenticated_username)"
  dataset_reference="$actor/$mil_training_dataset_slug"
  validate_mil_training "$actor"
  require_kaggle_cli
  if [[ -e "$mil_training_input_receipt" ]]; then
    echo "error: immutable Spec 0036 input receipt already exists" >&2
    exit 1
  fi
  stage_parent="$(mktemp -d "$TMPDIR/spec0036_input_verify.XXXXXX")"
  trap 'rm -rf -- "$stage_parent"' EXIT
  download_dir="$stage_parent/download"
  metadata_dir="$stage_parent/metadata"
  status_path="$stage_parent/status.json"
  mkdir -p "$download_dir" "$metadata_dir"
  kaggle_api datasets status "$dataset_reference" \
    --format 'json(status,current_version_number)' >"$status_path"
  dataset_version="$(python3 - "$status_path" <<'PYSPEC0036VERSION'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("status") != "ready":
    raise SystemExit("remote Spec 0036 input dataset is not ready")
version = status.get("current_version_number")
if isinstance(version, bool) or version != 1:
    raise SystemExit("remote Spec 0036 input dataset must be immutable version 1")
print(version)
PYSPEC0036VERSION
)"
  kaggle_api datasets metadata "$dataset_reference" -p "$metadata_dir"
  kaggle_api datasets download "$dataset_reference/$dataset_version" \
    -p "$download_dir" --unzip -o -q
  "$build_python" - \
    "$mil_training_root" "$download_dir" \
    "$metadata_dir/dataset-metadata.json" "$dataset_reference" \
    "$dataset_version" "$mil_training_input_receipt" <<'PYSPEC0036RECEIPT'
import hashlib
import json
import os
import sys
from pathlib import Path

root, downloaded_root, metadata_path, dataset_reference, version_text, receipt = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
    Path(sys.argv[6]),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


downloaded = {
    path.relative_to(downloaded_root).as_posix(): path
    for path in downloaded_root.rglob("*")
    if path.is_file() and path.name != "dataset-metadata.json"
}
bundle = root / "bundle"
expected_members = {
    path.relative_to(bundle).as_posix(): path
    for path in bundle.rglob("*")
    if path.is_file() and path.name != "dataset-metadata.json"
}
if set(downloaded) != set(expected_members):
    raise SystemExit("downloaded Spec 0036 input allow-list differs")
remote_files = []
for name, local_path in sorted(expected_members.items()):
    remote_path = downloaded[name]
    if (
        local_path.stat().st_size != remote_path.stat().st_size
        or sha256(local_path) != sha256(remote_path)
    ):
        raise SystemExit(f"downloaded Spec 0036 input differs: {name}")
    remote_files.append(
        {
            "logical_name": name,
            "bytes": local_path.stat().st_size,
            "sha256": sha256(local_path),
        }
    )
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
info = metadata.get("info", {})
owner, slug = dataset_reference.split("/", maxsplit=1)
if (
    not isinstance(info, dict)
    or info.get("ownerUser") != owner
    or info.get("datasetSlug") != slug
    or info.get("isPrivate") is not True
):
    raise SystemExit("remote Spec 0036 input identity/privacy differs")
contract = bundle / "mil_training_input.json"
record = {
    "schema_version": "spec0036.input_dataset_receipt.v1",
    "dataset_reference": dataset_reference,
    "dataset_version": int(version_text),
    "visibility": "private",
    "status": "verified",
    "input_contract_sha256": sha256(contract),
    "remote_files": remote_files,
}
receipt.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(receipt)
directory = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0036RECEIPT
  echo "ok: wrote verified immutable Spec 0036 input receipt $mil_training_input_receipt"
)

output_mil_training() {
  local receipt="${1:?launch receipt required}"
  local output_dir="${2:?output directory required}"
  local kernel_reference
  kernel_reference="$(kernel_reference_from_launch_receipt "$receipt")"
  if [[ ! "$kernel_reference" =~ ^[^/]+/eqvae-local-global-mil-training/[0-9]+$ ]]; then
    echo "error: launch receipt is not for Spec 0036 MIL training" >&2
    exit 1
  fi
  require_remote_confirmed
  require_kaggle_cli
  if [[ -e "$output_dir" ]]; then
    echo "error: output-mil-training requires a new output directory" >&2
    exit 1
  fi
  mkdir -p "$output_dir"
  kaggle_api kernels output "$kernel_reference" -p "$output_dir"
  record_kaggle_download \
    kernel "$kernel_reference" "$output_dir" kaggle_output_receipt.json
}

build_supervised_calibration_input() {
  local mode="$1"
  local selection_audit="${2:-}"
  if [[ "$mode" != "sweep" && "$mode" != "confirmation" && "$mode" != "horizon" && "$mode" != "width128" && "$mode" != "class_specific" && "$mode" != "class_specific_scale_fix" ]]; then
    echo "error: unknown calibration input mode" >&2
    exit 1
  fi
  if [[ "$mode" == "confirmation" && -z "$selection_audit" ]]; then
    echo "error: confirmation input requires its selection audit" >&2
    exit 1
  fi
  require_build_python
  local output_root="$supervised_calibration_input_root/$mode"
  local args=(
    -m eqvae.cli.build_ubc_supervised_calibration_inputs
    "$mode" build
    --repo-root "$PWD"
    --manifest-root runs/local/ubc_ocean_supervised_manifests
    --output-root "$output_root"
  )
  if [[ -n "$selection_audit" ]]; then
    args+=(--selection-audit "$selection_audit")
  fi
  "$build_python" "${args[@]}"
  echo "ok: built Spec 0023 $mode input bundle $output_root"
}

validate_supervised_calibration_input() {
  local mode="$1"
  local selection_audit="${2:-}"
  if [[ "$mode" != "sweep" && "$mode" != "confirmation" && "$mode" != "horizon" && "$mode" != "width128" && "$mode" != "class_specific" && "$mode" != "class_specific_scale_fix" ]]; then
    echo "error: unknown calibration input mode" >&2
    exit 1
  fi
  if [[ "$mode" == "confirmation" && -z "$selection_audit" ]]; then
    echo "error: confirmation input requires its selection audit" >&2
    exit 1
  fi
  require_build_python
  local args=(
    -m eqvae.cli.build_ubc_supervised_calibration_inputs
    "$mode" validate
    --repo-root "$PWD"
    --manifest-root runs/local/ubc_ocean_supervised_manifests
    --output-root "$supervised_calibration_input_root/$mode"
  )
  if [[ -n "$selection_audit" ]]; then
    args+=(--selection-audit "$selection_audit")
  fi
  "$build_python" "${args[@]}"
}

publish_supervised_calibration_input() (
  local mode="$1"
  local selection_audit="${2:-}"
  local receipt
  if [[ "$mode" == "sweep" ]]; then
    receipt="$supervised_calibration_sweep_receipt"
  elif [[ "$mode" == "confirmation" ]]; then
    receipt="$supervised_calibration_confirmation_receipt"
  elif [[ "$mode" == "horizon" ]]; then
    receipt="$supervised_calibration_horizon_receipt"
  elif [[ "$mode" == "width128" ]]; then
    receipt="$supervised_calibration_width128_receipt"
  elif [[ "$mode" == "class_specific" ]]; then
    receipt="$supervised_calibration_class_specific_receipt"
  elif [[ "$mode" == "class_specific_scale_fix" ]]; then
    receipt="$supervised_calibration_class_scale_receipt"
  else
    echo "error: unknown calibration input mode" >&2
    exit 1
  fi
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_SUPERVISED_CALIBRATION_CONFIRMED:-}" != "1" ]]; then
    echo "error: exact Kaggle dataset/calibration write confirmations are required" >&2
    exit 1
  fi
  if [[ -e "$receipt" ]]; then
    echo "error: immutable $mode input receipt already exists" >&2
    exit 1
  fi
  validate_supervised_calibration_input "$mode" "$selection_audit"
  require_kaggle_cli
  local stage_parent upload_dir create_output
  stage_parent="$(mktemp -d "$TMPDIR/spec0023_${mode}_input_upload.XXXXXX")"
  trap 'rm -rf -- "$stage_parent"' EXIT
  upload_dir="$stage_parent/envelope"
  local args=(
    -m eqvae.cli.build_ubc_supervised_calibration_inputs
    "$mode" stage-upload
    --repo-root "$PWD"
    --manifest-root runs/local/ubc_ocean_supervised_manifests
    --output-root "$supervised_calibration_input_root/$mode"
    --destination "$upload_dir"
  )
  if [[ -n "$selection_audit" ]]; then
    args+=(--selection-audit "$selection_audit")
  fi
  "$build_python" "${args[@]}"
  if ! create_output="$(kaggle_api datasets create -p "$upload_dir" 2>&1)"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  if [[ "$create_output" == *"Dataset creation error"* ]]; then
    echo "error: Kaggle reported dataset creation failure" >&2
    exit 1
  fi
)

verify_supervised_calibration_input() (
  local mode="$1"
  local selection_audit="${2:-}"
  local receipt dataset_slug
  if [[ "$mode" == "sweep" ]]; then
    receipt="$supervised_calibration_sweep_receipt"
    dataset_slug="$supervised_calibration_sweep_dataset_slug"
  elif [[ "$mode" == "confirmation" ]]; then
    receipt="$supervised_calibration_confirmation_receipt"
    dataset_slug="$supervised_calibration_confirmation_dataset_slug"
  elif [[ "$mode" == "horizon" ]]; then
    receipt="$supervised_calibration_horizon_receipt"
    dataset_slug="$supervised_calibration_horizon_dataset_slug"
  elif [[ "$mode" == "width128" ]]; then
    receipt="$supervised_calibration_width128_receipt"
    dataset_slug="$supervised_calibration_width128_dataset_slug"
  elif [[ "$mode" == "class_specific" ]]; then
    receipt="$supervised_calibration_class_specific_receipt"
    dataset_slug="$supervised_calibration_class_specific_dataset_slug"
  elif [[ "$mode" == "class_specific_scale_fix" ]]; then
    receipt="$supervised_calibration_class_scale_receipt"
    dataset_slug="$supervised_calibration_class_scale_dataset_slug"
  else
    echo "error: unknown calibration input mode" >&2
    exit 1
  fi
  require_remote_confirmed
  validate_supervised_calibration_input "$mode" "$selection_audit"
  require_kaggle_cli
  if [[ -e "$receipt" ]]; then
    echo "error: immutable $mode input receipt already exists" >&2
    exit 1
  fi
  local stage_parent download_dir metadata_dir status_path dataset_version
  stage_parent="$(mktemp -d "$TMPDIR/spec0023_${mode}_input_verify.XXXXXX")"
  trap 'rm -rf -- "$stage_parent"' EXIT
  download_dir="$stage_parent/download"
  metadata_dir="$stage_parent/metadata"
  status_path="$stage_parent/status.json"
  mkdir -p "$download_dir" "$metadata_dir"
  kaggle_api datasets status "$dataset_slug" \
    --format 'json(status,current_version_number)' >"$status_path"
  dataset_version="$(python3 - "$status_path" <<'PYVERSION'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("status") != "ready":
    raise ValueError("remote calibration input dataset is not ready")
version = status.get("current_version_number")
if isinstance(version, bool) or not isinstance(version, int) or version < 1:
    raise ValueError("remote calibration input dataset version is invalid")
print(version)
PYVERSION
)"
  kaggle_api datasets metadata "$dataset_slug" \
    -p "$metadata_dir"
  kaggle_api datasets download \
    "$dataset_slug/$dataset_version" \
    -p "$download_dir" --unzip -o -q
  "$build_python" - \
    "$supervised_calibration_input_root/$mode" \
    "$download_dir" \
    "$metadata_dir/dataset-metadata.json" \
    "$dataset_version" \
    "$receipt" \
    "$mode" <<'PYRECEIPT'
import sys
from pathlib import Path

from eqvae.cli.build_ubc_supervised_calibration_inputs import seal_remote_receipt

seal_remote_receipt(
    bundle_root=Path(sys.argv[1]),
    downloaded_root=Path(sys.argv[2]),
    remote_metadata_path=Path(sys.argv[3]),
    dataset_version=int(sys.argv[4]),
    output_path=Path(sys.argv[5]),
    package_mode=sys.argv[6],
)
PYRECEIPT
  echo "ok: wrote verified $mode input receipt $receipt"
)

build_supervised_calibration() {
  local mode="$1"
  local selection_audit="${2:-}"
  local sweep_audit="${3:-}"
  local sweep_config="${4:-}"
  local output_root="$supervised_calibration_root/$mode"
  require_build_python
  local args=(
    -m eqvae.cli.build_ubc_supervised_calibration "$mode" build
    --repo-root "$PWD"
    --manifest-root runs/local/ubc_ocean_supervised_manifests
    --output-root "$output_root"
  )
  if [[ -n "$selection_audit" ]]; then
    args+=(--selection-audit "$selection_audit")
  fi
  if [[ -n "$sweep_audit" ]]; then
    args+=(--sweep-audit "$sweep_audit")
  fi
  if [[ -n "$sweep_config" ]]; then
    args+=(--sweep-config "$sweep_config")
  fi
  "$build_python" "${args[@]}"
  preflight_supervised_calibration \
    "$mode" "$selection_audit" "$sweep_audit" "$sweep_config"
}

preflight_supervised_calibration() {
  local mode="$1"
  local selection_audit="${2:-}"
  local sweep_audit="${3:-}"
  local sweep_config="${4:-}"
  local output_root="$supervised_calibration_root/$mode"
  require_build_python
  local args=(
    -m eqvae.cli.build_ubc_supervised_calibration "$mode" validate
    --repo-root "$PWD"
    --manifest-root runs/local/ubc_ocean_supervised_manifests
    --output-root "$output_root"
  )
  if [[ -n "$selection_audit" ]]; then
    args+=(--selection-audit "$selection_audit")
  fi
  if [[ -n "$sweep_audit" ]]; then
    args+=(--sweep-audit "$sweep_audit")
  fi
  if [[ -n "$sweep_config" ]]; then
    args+=(--sweep-config "$sweep_config")
  fi
  "$build_python" "${args[@]}"
  if [[ "$mode" == "width128" ]]; then
    "$build_python" -m pytest -q \
      tests/test_spec0023_supervised_models.py::test_width128_changes_only_the_gated_scorer_width \
      tests/test_spec0023_supervised_calibration.py::test_width128_is_one_fresh_five_epoch_architecture_change \
      tests/test_spec0023_supervised_calibration.py::test_kaggle_kernel_id_and_title_fit_remote_limits
  elif [[ "$mode" == "class_specific" ]]; then
    "$build_python" -m pytest -q \
      tests/test_spec0023_supervised_models.py::test_class_specific_attention_starts_as_width128_then_can_specialize \
      tests/test_spec0023_supervised_calibration.py::test_class_specific_is_one_fresh_five_epoch_architecture_change \
      tests/test_spec0023_supervised_calibration.py::test_kaggle_kernel_id_and_title_fit_remote_limits
  elif [[ "$mode" == "class_specific_scale_fix" ]]; then
    "$build_python" -m pytest -q \
      tests/test_spec0023_supervised_models.py::test_class_specific_attention_starts_as_width128_then_can_specialize \
      tests/test_spec0023_supervised_calibration.py::test_class_specific_scale_fix_is_one_paired_amp_correction \
      tests/test_spec0023_supervised_calibration.py::test_kaggle_kernel_id_and_title_fit_remote_limits
  else
    "$build_python" -m pytest -q tests/test_spec0023_supervised_calibration.py
  fi
  echo "ok: Spec 0023 compact supervised calibration $mode preflight"
}

guard_supervised_calibration_push_ready() {
  local kernel_dir="$1"
  if [[ "$kernel_dir" != "$supervised_calibration_root/sweep" \
    && "$kernel_dir" != "$supervised_calibration_root/confirmation" \
    && "$kernel_dir" != "$supervised_calibration_root/horizon" \
    && "$kernel_dir" != "$supervised_calibration_root/width128" \
    && "$kernel_dir" != "$supervised_calibration_root/class_specific" \
    && "$kernel_dir" != "$supervised_calibration_root/class_specific_scale_fix" ]]; then
    echo "error: invalid Spec 0023 calibration push directory" >&2
    exit 1
  fi
  if [[ "${KAGGLE_SUPERVISED_CALIBRATION_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_SUPERVISED_CALIBRATION_CONFIRMED=1 after explicit authorization" >&2
    exit 1
  fi
  local mode="${kernel_dir##*/}"
  preflight_supervised_calibration \
    "$mode" \
    "${KAGGLE_SUPERVISED_SELECTION_AUDIT:-}" \
    "${KAGGLE_SUPERVISED_SWEEP_AUDIT:-}" \
    "${KAGGLE_SUPERVISED_SWEEP_CONFIG:-}"
}

publish_cancer_topup_inputs() (
  require_build_python
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_CANCER_TOPUP_CONFIRMED:-}" != "1" ]]; then
    echo "error: exact Kaggle dataset/top-up write confirmations are required" >&2
    exit 1
  fi
  if [[ -e "$cancer_topup_receipt" ]]; then
    echo "error: immutable Spec 0022 input receipt already exists" >&2
    exit 1
  fi
  "$build_python" -m eqvae.cli.build_ubc_cancer_topup --validate-only
  require_kaggle_cli
  local create_output status_output
  if status_output="$(kaggle_api datasets status \
    "maximusshtefan/eqvae-ubc-ocean-cancer-topup-inputs" \
    --format 'json(status,current_version_number)' 2>&1)"; then
    if ! create_output="$(kaggle_api datasets version \
      -p "$cancer_topup_plan_root/inference_bundle" \
      -m "Spec 0022 flat checkpoint input bundle" 2>&1)"; then
      printf '%s\n' "$create_output" >&2
      exit 1
    fi
  elif [[ "$status_output" == *"404"* \
    || "$status_output" == *"Not Found"* \
    || "$status_output" == *"not found"* ]]; then
    if ! create_output="$(kaggle_api datasets create \
      -p "$cancer_topup_plan_root/inference_bundle" 2>&1)"; then
      printf '%s\n' "$create_output" >&2
      exit 1
    fi
  else
    printf '%s\n' "$status_output" >&2
    echo "error: refusing to create after an ambiguous Kaggle status failure" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  if [[ "$create_output" == *"Dataset creation error"* ]]; then
    echo "error: Kaggle reported Spec 0022 input creation failure" >&2
    exit 1
  fi
)

verify_cancer_topup_inputs() (
  require_build_python
  require_remote_confirmed
  "$build_python" -m eqvae.cli.build_ubc_cancer_topup --validate-only
  require_kaggle_cli
  local verify_root download_dir metadata_dir status_path listing_path version
  verify_root="$(mktemp -d "$TMPDIR/spec0022_input_verify.XXXXXX")"
  trap 'rm -rf -- "$verify_root"' EXIT
  download_dir="$verify_root/download"
  metadata_dir="$verify_root/metadata"
  status_path="$verify_root/status.json"
  listing_path="$verify_root/files.csv"
  mkdir -p "$download_dir" "$metadata_dir"
  kaggle_api datasets status "maximusshtefan/eqvae-ubc-ocean-cancer-topup-inputs" \
    --format 'json(status,current_version_number)' >"$status_path"
  version="$(python3 - "$status_path" <<'PYSPEC0022VERSION'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
version = value.get("current_version_number")
if value.get("status") != "ready" or isinstance(version, bool) or not isinstance(version, int) or version < 1:
    raise ValueError("remote Spec 0022 input dataset is not ready")
print(version)
PYSPEC0022VERSION
)"
  local reference="maximusshtefan/eqvae-ubc-ocean-cancer-topup-inputs/$version"
  kaggle_api datasets files "$reference" --csv >"$listing_path"
  kaggle_api datasets metadata \
    "maximusshtefan/eqvae-ubc-ocean-cancer-topup-inputs" -p "$metadata_dir"
  kaggle_api datasets download "$reference" -p "$download_dir" --unzip -o -q
  python3 - \
    "$cancer_topup_plan_root/inference_bundle" \
    "$download_dir" \
    "$metadata_dir/dataset-metadata.json" \
    "$status_path" \
    "$listing_path" \
    "$version" \
    "$cancer_topup_receipt" <<'PYSPEC0022RECEIPT'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

local_root = Path(sys.argv[1])
download_root = Path(sys.argv[2])
metadata_path = Path(sys.argv[3])
status_path = Path(sys.argv[4])
listing_path = Path(sys.argv[5])
version = int(sys.argv[6])
receipt_path = Path(sys.argv[7])
if receipt_path.exists():
    raise FileExistsError(f"refusing to overwrite {receipt_path}")
local = {
    path.relative_to(local_root).as_posix(): path
    for path in local_root.rglob("*")
    if path.is_file() and path.name != "dataset-metadata.json"
}
downloaded = {
    path.relative_to(download_root).as_posix(): path
    for path in download_root.rglob("*")
    if path.is_file()
}
if set(downloaded) != set(local):
    raise ValueError("downloaded Spec 0022 input allow-list differs")
files = {}
for name, path in sorted(local.items()):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    remote = downloaded[name]
    if remote.stat().st_size != path.stat().st_size or hashlib.sha256(remote.read_bytes()).hexdigest() != digest:
        raise ValueError(f"downloaded Spec 0022 input differs: {name}")
    files[name] = {"bytes": path.stat().st_size, "sha256": digest}
with listing_path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fields = {name.casefold(): name for name in (reader.fieldnames or ())}
    if "name" not in fields or "size" not in fields:
        raise ValueError("remote Spec 0022 listing lacks name/size")
    listed = {str(row[fields["name"]]): int(str(row[fields["size"]])) for row in reader}
if listed != {name: path.stat().st_size for name, path in local.items()}:
    raise ValueError("remote Spec 0022 file listing differs")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
info = metadata.get("info")
if not isinstance(info, dict) or info.get("ownerUser") != "maximusshtefan" or info.get("datasetSlug") != "eqvae-ubc-ocean-cancer-topup-inputs" or info.get("isPrivate") is not True:
    raise ValueError("remote Spec 0022 identity/privacy differs")
contract = local_root / "spec0022_topup_inference_contract.json"
payload = {
    "schema_version": "spec0022.input_dataset_receipt.v1",
    "status": "verified",
    "visibility": "private",
    "dataset_reference": "maximusshtefan/eqvae-ubc-ocean-cancer-topup-inputs",
    "dataset_version": version,
    "input_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "files": files,
    "remote_listing_sha256": hashlib.sha256(listing_path.read_bytes()).hexdigest(),
    "remote_status_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
}
encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
receipt_path.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt_path.with_suffix(".json.tmp")
with temporary.open("xb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, receipt_path)
directory = os.open(receipt_path.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYSPEC0022RECEIPT
  echo "ok: wrote verified immutable Spec 0022 input receipt $cancer_topup_receipt"
)

guard_latent_inference_push_ready() {
  local kernel_dir="$1"
  if python3 - "$kernel_dir/spec0021_inference_config.json" <<'PYFINALIZERMODE'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(
    0 if config.get("schema_version") == "spec0021.finalizer_config.v1" else 1
)
PYFINALIZERMODE
  then
    validate_latent_finalizer_dir "$kernel_dir"
    return
  fi
  local receipt_policy="fresh"
  if python3 - "$kernel_dir/spec0021_inference_config.json" <<'PYRESUMEMODE'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if config.get("resume_dataset_receipt") is not None else 1)
PYRESUMEMODE
  then
    receipt_policy="resume"
  fi
  validate_latent_kernel_dir "$kernel_dir" "$receipt_policy"
}

validate_latent_input_bundle() {
  require_build_python
  "$build_python" - "$latent_input_bundle_dir" "$latent_input_dataset_slug" <<'PYBUNDLE'
import sys
from pathlib import Path

from eqvae.inference.input_bundle import validate_fresh_input_bundle

validate_fresh_input_bundle(
    Path(sys.argv[1]),
    expected_dataset_slug=sys.argv[2],
)
PYBUNDLE
}

stage_latent_input_upload() {
  local destination="$1"
  require_build_python
  "$build_python" - \
    "$latent_input_bundle_dir" \
    "$destination" \
    "$latent_input_dataset_slug" <<'PYSTAGEUPLOAD'
import sys
from pathlib import Path

from eqvae.inference.input_bundle import stage_fresh_upload_archive

stage_fresh_upload_archive(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    expected_dataset_slug=sys.argv[3],
)
PYSTAGEUPLOAD
}

publish_latent_inputs() (
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_PUSH_CONFIRMED=1 after explicit user permission" >&2
    exit 1
  fi
  if [[ "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_DATASET_WRITE_CONFIRMED=1 for this exact input publication" >&2
    exit 1
  fi
  if [[ -e "$latent_input_receipt" ]]; then
    echo "error: input dataset receipt already exists; immutable inputs are not republished" >&2
    exit 1
  fi
  validate_latent_input_bundle
  require_kaggle_cli
  local stage_parent upload_dir create_output
  stage_parent="$(mktemp -d "$TMPDIR/spec0021_input_upload.XXXXXX")"
  trap 'rm -rf -- "$stage_parent"' EXIT
  upload_dir="$stage_parent/envelope"
  stage_latent_input_upload "$upload_dir"
  # Kaggle datasets are private by default; current CLI uses -u only for public.
  if ! create_output="$(kaggle_api datasets create -p "$upload_dir" 2>&1)"; then
    printf '%s\n' "$create_output" >&2
    exit 1
  fi
  printf '%s\n' "$create_output"
  if [[ "$create_output" == *"Dataset creation error"* ]]; then
    echo "error: Kaggle CLI reported dataset creation failure with a zero exit status" >&2
    exit 1
  fi
)

verify_latent_inputs() (
  require_remote_confirmed
  validate_latent_input_bundle
  require_kaggle_cli
  local stage_parent download_dir metadata_dir status_path dataset_version
  local versioned_reference
  stage_parent="$(mktemp -d "$TMPDIR/spec0021_input_verify.XXXXXX")"
  trap 'rm -rf -- "$stage_parent"' EXIT
  download_dir="$stage_parent/download"
  metadata_dir="$stage_parent/metadata"
  status_path="$stage_parent/status.json"
  mkdir -p "$download_dir"
  mkdir -p "$metadata_dir"
  kaggle_api datasets status "$latent_input_dataset_slug" \
    --format 'json(status,current_version_number)' >"$status_path"
  dataset_version="$(python3 - "$status_path" <<'PYINPUTVERSION'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("status") != "ready":
    raise ValueError("remote input dataset is not ready")
version = status.get("current_version_number")
if isinstance(version, bool) or not isinstance(version, int) or version < 1:
    raise ValueError("remote input dataset version is invalid")
print(version)
PYINPUTVERSION
)"
  versioned_reference="$latent_input_dataset_slug/$dataset_version"
  local listing_path="$TMPDIR/spec0021_input_remote_files.csv"
  kaggle_api datasets files "$versioned_reference" --csv >"$listing_path"
  kaggle_api datasets metadata "$latent_input_dataset_slug" -p "$metadata_dir"
  kaggle_api datasets download "$versioned_reference" \
    -p "$download_dir" --unzip -o -q
  mkdir -p "$latent_input_authority_dir"
  python3 - \
    "$latent_input_bundle_dir" \
    "$download_dir" \
    "$metadata_dir/dataset-metadata.json" \
    "$latent_input_dataset_slug" \
    "$dataset_version" \
    "$status_path" \
    "$listing_path" \
    "$latent_input_receipt" <<'PYRECEIPT'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
downloaded = Path(sys.argv[2])
remote_metadata_path = Path(sys.argv[3])
dataset_slug = sys.argv[4]
dataset_version = int(sys.argv[5])
status_path = Path(sys.argv[6])
listing_path = Path(sys.argv[7])
receipt_path = Path(sys.argv[8])
if receipt_path.exists():
    raise FileExistsError(f"refusing to overwrite {receipt_path}")
with listing_path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise ValueError("remote dataset listing has no header")
    field_map = {name.casefold(): name for name in reader.fieldnames}
    name_field = field_map.get("name")
    size_field = field_map.get("size")
    if name_field is None or size_field is None:
        raise ValueError("remote dataset listing must contain name and size")
    remote = {
        str(row[name_field]): int(str(row[size_field]))
        for row in reader
    }
local = {
    path.relative_to(bundle).as_posix(): path.stat().st_size
    for path in bundle.rglob("*")
    if path.is_file()
}
if remote != local:
    raise ValueError("remote input dataset listing differs from local bundle")
downloaded_files = {
    path.relative_to(downloaded).as_posix(): path
    for path in downloaded.rglob("*")
    if path.is_file()
}
if set(downloaded_files) != set(local):
    raise ValueError("downloaded input dataset allow-list differs")
file_sha256 = {}
for logical_name in sorted(local):
    local_path = bundle / logical_name
    downloaded_path = downloaded_files[logical_name]
    local_sha256 = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if (
        downloaded_path.stat().st_size != local[logical_name]
        or hashlib.sha256(downloaded_path.read_bytes()).hexdigest() != local_sha256
    ):
        raise ValueError(f"downloaded input file differs: {logical_name}")
    file_sha256[logical_name] = local_sha256
remote_metadata = json.loads(remote_metadata_path.read_text(encoding="utf-8"))
info = remote_metadata.get("info")
owner, slug = dataset_slug.split("/", maxsplit=1)
if (
    not isinstance(info, dict)
    or info.get("ownerUser") != owner
    or info.get("datasetSlug") != slug
    or info.get("isPrivate") is not True
):
    raise ValueError("remote input dataset identity or privacy differs")
contract = bundle / "spec0021_input_contract.json"
payload = {
    "schema_version": "spec0021.input_dataset_receipt.v1",
    "dataset_reference": dataset_slug,
    "dataset_version": dataset_version,
    "input_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "remote_files": [
        {"logical_name": name, "sha256": file_sha256[name], "size": size}
        for name, size in sorted(remote.items())
    ],
    "remote_listing_sha256": hashlib.sha256(listing_path.read_bytes()).hexdigest(),
    "remote_status_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
}
encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
temporary = receipt_path.with_suffix(".json.tmp")
with temporary.open("xb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, receipt_path)
directory = os.open(receipt_path.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYRECEIPT
  echo "ok: wrote verified immutable input receipt $latent_input_receipt"
)

latest_latent_resume_stage() {
  local run_number="$1"
  python3 - "$latent_resume_root" "$run_number" <<'PYLATESTRESUME'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) / f"run_{sys.argv[2]}"
versions = [
    (int(match.group(1)), path)
    for path in root.glob("version_*")
    if path.is_dir()
    and (match := re.fullmatch(r"version_([0-9]{4})", path.name)) is not None
]
if not versions:
    raise FileNotFoundError(f"no staged resume bundle under {root}")
print(max(versions)[1])
PYLATESTRESUME
}

validate_latent_resume_stage() {
  local run_number="$1"
  local stage_dir="$2"
  require_build_python
  "$build_python" - \
    "$stage_dir" \
    "$latent_input_bundle_dir/manifests/work_shards/run_${run_number}_of_05.csv" <<'PYVALIDATERESUME'
import hashlib
import json
import sys
from pathlib import Path

from eqvae.inference.input_bundle import (
    RESUME_PROVENANCE_FILENAME,
    ResumeBundleAuthority,
    validate_resume_bundle,
)

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
contract_path = root / RESUME_PROVENANCE_FILENAME
payload = json.loads(contract_path.read_text(encoding="utf-8"))
dataset = payload["dataset"]
authority = ResumeBundleAuthority(
    provenance_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    dataset_slug=dataset["slug"],
    dataset_version=dataset["version"],
    run_number=payload["run_number"],
    input_bundle_sha256=payload["input_bundle_sha256"],
    run_config_sha256=payload["run_config_sha256"],
    work_manifest_sha256=payload["work_manifest_sha256"],
)
validate_resume_bundle(root, authority=authority, work_manifest_path=manifest)
PYVALIDATERESUME
}

build_latent_resume() {
  local run_number
  run_number="$(normalize_latent_run_number "${1:-}")"
  local artifacts_dir="${2:-}"
  if [[ -z "$artifacts_dir" ]]; then
    echo "error: build-latent-resume requires XX and artifacts-dir" >&2
    exit 1
  fi
  local run_config="$latent_inference_kernel_root/run_${run_number}/spec0021_inference_config.json"
  if [[ ! -f "$run_config" ]]; then
    echo "error: missing run config $run_config" >&2
    exit 1
  fi
  require_build_python
  "$build_python" -m eqvae.cli.stage_ubc_latent_resume \
    "$((10#$run_number))" "$artifacts_dir" \
    --input-contract "$latent_input_bundle_dir/spec0021_input_contract.json" \
    --run-config "$run_config" \
    --prior-receipt \
      "$latent_input_authority_dir/resume_run_${run_number}_dataset_receipt.json" \
    --output-root "$latent_resume_root"
  local stage_dir
  stage_dir="$(latest_latent_resume_stage "$run_number")"
  validate_latent_resume_stage "$run_number" "$stage_dir"
  echo "ok: staged Spec 0021 run-$run_number resume bundle at $stage_dir"
}

publish_latent_resume() {
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_PUSH_CONFIRMED=1 after explicit user permission" >&2
    exit 1
  fi
  if [[ "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_DATASET_WRITE_CONFIRMED=1 for this exact resume publication" >&2
    exit 1
  fi
  local run_number
  run_number="$(normalize_latent_run_number "${1:-}")"
  local stage_dir
  stage_dir="$(latest_latent_resume_stage "$run_number")"
  validate_latent_resume_stage "$run_number" "$stage_dir"
  local dataset_version
  dataset_version="$(python3 - "$stage_dir/spec0021_resume_contract.json" <<'PYRESUMEVERSION'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["dataset"]["version"])
PYRESUMEVERSION
)"
  python3 - \
    "$stage_dir/spec0021_resume_contract.json" \
    "$latent_input_authority_dir/resume_run_${run_number}_dataset_receipt.json" <<'PYPUBLISHRESUME'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
version = contract["dataset"]["version"]
receipt_path = Path(sys.argv[2])
if receipt_path.is_file():
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("dataset_version") != version - 1:
        raise ValueError("staged resume version is not after the verified receipt")
elif version != 1:
    raise ValueError("resume dataset version above 1 requires a prior receipt")
PYPUBLISHRESUME
  require_kaggle_cli
  if [[ "$dataset_version" == "1" ]]; then
    # Kaggle datasets are private by default; current CLI uses -u only for public.
    kaggle_api datasets create -p "$stage_dir"
  else
    kaggle_api datasets version -p "$stage_dir" \
      -m "Spec 0021 run-$run_number immutable resume version $dataset_version"
  fi
}

verify_latent_resume() {
  require_remote_confirmed
  local run_number
  run_number="$(normalize_latent_run_number "${1:-}")"
  local stage_dir
  stage_dir="$(latest_latent_resume_stage "$run_number")"
  validate_latent_resume_stage "$run_number" "$stage_dir"
  require_kaggle_cli
  local dataset_slug="maximusshtefan/eqvae-ubc-ocean-latent-run-${run_number}-resume"
  local listing_path="$TMPDIR/spec0021_resume_run_${run_number}_remote_files.csv"
  local receipt_path="$latent_input_authority_dir/resume_run_${run_number}_dataset_receipt.json"
  kaggle_api datasets files "$dataset_slug" --csv >"$listing_path"
  mkdir -p "$latent_input_authority_dir"
  python3 - "$stage_dir" "$dataset_slug" "$listing_path" "$receipt_path" <<'PYRESUMERECEIPT'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
dataset_slug = sys.argv[2]
listing_path = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])
contract_path = bundle / "spec0021_resume_contract.json"
contract = json.loads(contract_path.read_text(encoding="utf-8"))
dataset = contract["dataset"]
if dataset != {"slug": dataset_slug, "version": dataset["version"]}:
    raise ValueError("staged resume dataset identity mismatch")
version = dataset["version"]
if receipt_path.is_file():
    prior = json.loads(receipt_path.read_text(encoding="utf-8"))
    if prior.get("dataset_reference") != dataset_slug:
        raise ValueError("prior resume receipt dataset mismatch")
    if prior.get("dataset_version") != version - 1:
        raise ValueError("resume dataset version is not the next immutable version")
elif version != 1:
    raise ValueError("first verified resume receipt must be dataset version 1")
with listing_path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise ValueError("remote resume listing has no header")
    fields = {name.casefold(): name for name in reader.fieldnames}
    if "name" not in fields or "size" not in fields:
        raise ValueError("remote resume listing must contain name and size")
    remote = {
        str(row[fields["name"]]): int(str(row[fields["size"]]))
        for row in reader
    }
local = {
    path.relative_to(bundle).as_posix(): path.stat().st_size
    for path in bundle.rglob("*")
    if path.is_file()
}
if remote != local:
    raise ValueError("remote resume dataset listing differs from staged bundle")
payload = {
    "schema_version": "spec0021.resume_dataset_receipt.v1",
    "dataset_reference": dataset_slug,
    "dataset_version": version,
    "run_number": contract["run_number"],
    "provenance_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "input_bundle_sha256": contract["input_bundle_sha256"],
    "run_config_sha256": contract["run_config_sha256"],
    "work_manifest_sha256": contract["work_manifest_sha256"],
    "remote_files": [
        {"logical_name": name, "size": size}
        for name, size in sorted(remote.items())
    ],
    "remote_listing_sha256": hashlib.sha256(listing_path.read_bytes()).hexdigest(),
}
encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
temporary = receipt_path.with_suffix(".json.tmp")
with temporary.open("xb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, receipt_path)
directory = os.open(receipt_path.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PYRESUMERECEIPT
  echo "ok: wrote verified resume receipt $receipt_path"
}

full_foreground_package() {
  require_build_python
  "$build_python" -m eqvae.cli.build_ubc_full_foreground_completion "$@"
}

publish_full_foreground_inputs() {
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_FULL_FOREGROUND_CONFIRMED:-}" != "1" ]]; then
    echo "error: exact full-foreground input publication permission is required" >&2
    exit 1
  fi
  full_foreground_package validate
  if [[ -e runs/local/full_foreground_completion/input_receipt.json ]]; then
    echo "error: immutable full-foreground input receipt already exists" >&2
    exit 1
  fi
  require_kaggle_cli
  kaggle_api datasets create -p runs/local/full_foreground_completion/upload
}

verify_full_foreground_inputs() {
  require_remote_confirmed
  full_foreground_package validate
  require_kaggle_cli
  local root="runs/local/full_foreground_completion/remote_v1"
  local slug="maximusshtefan/eqvae-full-foreground-inputs"
  mkdir -p "$root/download" "$root/metadata"
  kaggle_api datasets status "$slug" --format 'json(status,current_version_number)' > "$root/status.json"
  kaggle_api datasets metadata "$slug" -p "$root/metadata"
  kaggle_api datasets download "$slug/1" -p "$root/download" --unzip -q
  full_foreground_package verify-download
}

output_full_foreground() {
  local number="${1:-}"
  if [[ ! "$number" =~ ^0[1-8]$ ]]; then
    echo "error: full-foreground output requires run 01 through 08" >&2
    exit 1
  fi
  require_remote_confirmed
  require_kaggle_cli
  local root="runs/kaggle/full_foreground_completion/run_$number"
  mkdir -p "$root"
  kaggle_api kernels output "maximusshtefan/eqvae-full-foreground-$number" \
    -p "$root" --file-pattern '.*\.(json|log)$'
}

check_wsi45630_package() {
  require_build_python
  "$build_python" - "${1:-check}" <<'PYWSIPACKAGE'
import hashlib
import json
import sys
from pathlib import Path
from string import Template

from eqvae.cli.build_ubc_wsi45630_completion import validate

root = Path("runs/local/wsi45630_completion")
validate(repo_root=Path.cwd(), output_root=root)
contract = root / "bundle/wsi45630_input.json"
digest = hashlib.sha256(contract.read_bytes()).hexdigest()
template = Path("kaggle/kernels/wsi45630_completion/run_template.py").read_text()
code = Template(template).substitute(input_contract_sha256=digest).encode()
assert len(code) < 1_000_000
compile(code, "run.py", "exec")
metadata = {
    "id": "maximusshtefan/eqvae-wsi45630-completion",
    "title": "eqvae wsi45630 completion",
    "code_file": "run.py", "language": "python", "kernel_type": "script",
    "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["maximusshtefan/eqvae-wsi45630-completion-inputs"],
    "competition_sources": ["UBC-OCEAN"], "kernel_sources": [], "model_sources": [],
}
kernel = root / "kernel"
if sys.argv[1] == "render":
    kernel.mkdir()
    (kernel / "run.py").write_bytes(code)
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
assert (kernel / "run.py").read_bytes() == code
assert json.loads((kernel / "kernel-metadata.json").read_text()) == metadata
if sys.argv[1] == "receipt":
    receipt = json.loads((root / "input_receipt.json").read_text())
    assert receipt["dataset_reference"] == metadata["dataset_sources"][0]
    assert receipt["dataset_version"] == 1 and receipt["visibility"] == "private"
    assert receipt["status"] == "verified" and receipt["input_contract_sha256"] == digest
    assert receipt["files"] == json.loads(contract.read_text())["files"]
print(f"ok: WSI45630 package, source={len(code)} bytes, mode={sys.argv[1]}")
PYWSIPACKAGE
}

build_wsi45630_package() {
  require_build_python
  "$build_python" -m eqvae.cli.build_ubc_wsi45630_completion \
    --repo-root "$PWD" --output-root runs/local/wsi45630_completion
  check_wsi45630_package render
}

publish_wsi45630_inputs() {
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_WSI45630_COMPLETION_CONFIRMED:-}" != "1" ]]; then
    echo "error: exact WSI45630 dataset publication permission is required" >&2
    exit 1
  fi
  check_wsi45630_package
  if [[ -e runs/local/wsi45630_completion/input_receipt.json ]]; then
    echo "error: immutable WSI45630 input receipt already exists" >&2
    exit 1
  fi
  require_kaggle_cli
  kaggle_api datasets create -p runs/local/wsi45630_completion/upload
}

verify_wsi45630_inputs() {
  require_remote_confirmed
  check_wsi45630_package
  require_kaggle_cli
  local root="runs/local/wsi45630_completion"
  local slug="maximusshtefan/eqvae-wsi45630-completion-inputs"
  mkdir -p "$root/remote_v1/download" "$root/remote_v1/metadata"
  kaggle_api datasets status "$slug" --format 'json(status,current_version_number)' > "$root/remote_v1/status.json"
  kaggle_api datasets metadata "$slug" -p "$root/remote_v1/metadata"
  kaggle_api datasets download "$slug/1" -p "$root/remote_v1/download" --unzip -q
  "$build_python" - <<'PYWSIRECEIPT'
import hashlib
import json
from pathlib import Path

root = Path("runs/local/wsi45630_completion")
status = json.loads((root / "remote_v1/status.json").read_text())
info = json.loads((root / "remote_v1/metadata/dataset-metadata.json").read_text())["info"]
assert status == {"status": "ready", "current_version_number": 1}
assert info["ownerUser"] == "maximusshtefan"
assert info["datasetSlug"] == "eqvae-wsi45630-completion-inputs" and info["isPrivate"] is True
bundle = root / "bundle"
contract = json.loads((bundle / "wsi45630_input.json").read_text())
names = {*contract["files"], "wsi45630_input.json"}
download = root / "remote_v1/download"
assert {p.relative_to(download).as_posix() for p in download.rglob("*") if p.is_file()} == names
for name in names:
    local, remote = bundle / name, download / name
    assert local.stat().st_size == remote.stat().st_size
    assert hashlib.sha256(local.read_bytes()).digest() == hashlib.sha256(remote.read_bytes()).digest()
receipt = {
    "status": "verified", "visibility": "private", "dataset_version": 1,
    "dataset_reference": contract["dataset_reference"], "files": contract["files"],
    "input_contract_sha256": hashlib.sha256((bundle / "wsi45630_input.json").read_bytes()).hexdigest(),
}
with (root / "input_receipt.json").open("x") as handle:
    json.dump(receipt, handle, indent=2)
    handle.write("\n")
print("ok: WSI45630 private input version 1 is byte-verified")
PYWSIRECEIPT
}

check_full_wsi_capacity() {
  require_build_python
  "$build_python" - "${1:-check}" <<'PYFULLWSICHECK'
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from build_wsi45630_capacity import validate

contract = validate()
root = Path("runs/local/wsi45630_capacity")
digest = hashlib.sha256((root / "bundle/wsi45630_capacity_input.json").read_bytes()).hexdigest()
if sys.argv[1] == "receipt":
    receipt = json.loads((root / "input_receipt.json").read_text())
    assert receipt["status"] == "verified" and receipt["visibility"] == "private"
    assert receipt["dataset_version"] == 1
    assert receipt["dataset_reference"] == contract["dataset_reference"]
    assert receipt["input_contract_sha256"] == digest
    assert receipt["files"] == contract["files"]
print("ok: full-WSI capacity package " + sys.argv[1])
PYFULLWSICHECK
}

publish_full_wsi_capacity_inputs() {
  if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" \
    || "${KAGGLE_DATASET_WRITE_CONFIRMED:-}" != "1" \
    || "${KAGGLE_WSI45630_CAPACITY_CONFIRMED:-}" != "1" ]]; then
    echo "error: full-WSI capacity dataset publication permission required" >&2
    exit 1
  fi
  check_full_wsi_capacity
  if [[ -e runs/local/wsi45630_capacity/input_receipt.json ]]; then
    echo "error: immutable full-WSI capacity input receipt already exists" >&2
    exit 1
  fi
  require_kaggle_cli
  kaggle_api datasets create -p runs/local/wsi45630_capacity/upload
}

verify_full_wsi_capacity_inputs() {
  require_remote_confirmed
  check_full_wsi_capacity
  require_kaggle_cli
  local root="runs/local/wsi45630_capacity"
  local slug="maximusshtefan/eqvae-wsi45630-capacity-inputs"
  mkdir -p "$root/remote_v1/download" "$root/remote_v1/metadata"
  kaggle_api datasets status "$slug" --format 'json(status,current_version_number)' > "$root/remote_v1/status.json"
  kaggle_api datasets metadata "$slug" -p "$root/remote_v1/metadata"
  kaggle_api datasets download "$slug/1" -p "$root/remote_v1/download" --unzip -q
  "$build_python" - <<'PYFULLWSIRECEIPT'
import hashlib
import json
from pathlib import Path

root = Path("runs/local/wsi45630_capacity")
status = json.loads((root / "remote_v1/status.json").read_text())
info = json.loads((root / "remote_v1/metadata/dataset-metadata.json").read_text())["info"]
assert status == {"status": "ready", "current_version_number": 1}
assert info["ownerUser"] == "maximusshtefan" and info["isPrivate"] is True
assert info["datasetSlug"] == "eqvae-wsi45630-capacity-inputs"
bundle = root / "bundle"
contract = json.loads((bundle / "wsi45630_capacity_input.json").read_text())
names = {*contract["files"], "wsi45630_capacity_input.json"}
download = root / "remote_v1/download"
assert {p.relative_to(download).as_posix() for p in download.rglob("*") if p.is_file()} == names
for name in names:
    local, remote = bundle / name, download / name
    assert local.stat().st_size == remote.stat().st_size
    assert hashlib.sha256(local.read_bytes()).digest() == hashlib.sha256(remote.read_bytes()).digest()
receipt = {
    "status": "verified", "visibility": "private", "dataset_version": 1,
    "dataset_reference": contract["dataset_reference"], "files": contract["files"],
    "input_contract_sha256": hashlib.sha256((bundle / "wsi45630_capacity_input.json").read_bytes()).hexdigest(),
}
with (root / "input_receipt.json").open("x") as handle:
    json.dump(receipt, handle, indent=2)
    handle.write("\n")
print("ok: private full-WSI capacity input version 1 is byte-verified")
PYFULLWSIRECEIPT
}

action="${1:-}"
case "$action" in
  build)
    kernel_dir="${2:-$default_kernel_dir}"
    if [[ "$kernel_dir" == "$ubc_ocean_test_atlas_kernel_dir" ]]; then
      build_ubc_ocean_test_atlas_kernel "$kernel_dir"
    elif [[ -f "$kernel_dir/run_template.py" ]]; then
      build_embedded_kernel "$kernel_dir"
    elif is_setup_kernel_dir "$kernel_dir"; then
      build_embedded_setup_kernel "$kernel_dir"
    else
      build_kernel_payload "$kernel_dir"
    fi
    ;;
  validate)
    validate_kernel_dir "${2:-$default_kernel_dir}"
    ;;
  check)
    validate_kernel_dir "${2:-$default_kernel_dir}"
    require_kaggle_cli
    kaggle --version
    ;;
  api-check)
    api_check "${2:-$default_kernel_dir}"
    ;;
  preflight-runtime-selection)
    preflight_runtime_selection
    ;;
  preflight-fixed32-selector-readiness)
    preflight_fixed32_selector_readiness
    ;;
  preflight-selected-runtime-runner)
    preflight_selected_runtime_runner
    ;;
  preflight-selected-runtime-debug)
    preflight_selected_runtime_debug
    ;;
  preflight-selected-runtime-lr-range)
    preflight_selected_runtime_lr_range
    ;;
  preflight-selected-runtime-full)
    preflight_selected_runtime_full
    ;;
  preflight-fixed25-selector)
    preflight_fixed25_selector
    ;;
  preflight-so2-architecture-probe)
    preflight_so2_architecture_probe
    ;;
  preflight-so2-runtime-readiness)
    preflight_so2_runtime_readiness
    ;;
  preflight-so2-prelaunch)
    preflight_so2_prelaunch
    ;;
  preflight-so2-selected-runtime-full)
    preflight_so2_full
    ;;
  build-latent-inference)
    build_latent_inference "${2:-}" "${3:-}"
    ;;
  preflight-latent-inference)
    preflight_latent_inference "${2:-}"
    ;;
  build-cancer-topup)
    build_cancer_topup
    ;;
  preflight-cancer-topup)
    preflight_cancer_topup
    ;;
  preflight-mil-capacity-probe)
    preflight_mil_capacity_probe
    ;;
  build-tissue-training)
    build_tissue_training "${2:-}"
    ;;
  validate-tissue-training)
    validate_tissue_training "${2:-}"
    ;;
  preflight-tissue-training)
    preflight_tissue_training "${2:-}"
    ;;
  build-tissue-training-retry-v2)
    build_tissue_training_retry "${2:?actor required}" "${3:?frozen input bundle required}"
    ;;
  preflight-tissue-training-retry-v2)
    preflight_tissue_training_retry "${2:-}"
    ;;
  build-tissue-training-retry-v3)
    build_tissue_training_retry_v3 "${2:?actor required}" "${3:?frozen input bundle required}"
    ;;
  preflight-tissue-training-retry-v3)
    preflight_tissue_training_retry_v3 "${2:-}"
    ;;
  publish-tissue-training-inputs)
    publish_tissue_training_inputs
    ;;
  verify-tissue-training-inputs)
    verify_tissue_training_inputs
    ;;
  output-tissue-training)
    output_tissue_training "${2:-}" "${3:-}"
    ;;
  build-tissue-fastpath-probe)
    build_tissue_fastpath_probe "${2:-}"
    ;;
  validate-tissue-fastpath-probe)
    validate_tissue_fastpath_probe "${2:-}"
    ;;
  preflight-tissue-fastpath-probe)
    preflight_tissue_fastpath_probe "${2:-}"
    ;;
  publish-tissue-fastpath-probe-inputs)
    publish_tissue_fastpath_probe_inputs
    ;;
  verify-tissue-fastpath-probe-inputs)
    verify_tissue_fastpath_probe_inputs
    ;;
  build-mil-training)
    build_mil_training "${2:-}"
    ;;
  build-mil-test)
    build_mil_test "${2:-}"
    ;;
  build-tissue-test)
    build_tissue_test "${2:-}"
    ;;
  validate-tissue-test)
    validate_tissue_test "${2:-}"
    ;;
  publish-tissue-test-inputs)
    publish_tissue_test_inputs
    ;;
  status-tissue-test-inputs)
    status_tissue_test_inputs
    ;;
  verify-tissue-test-inputs)
    verify_tissue_test_inputs
    ;;
  push-tissue-test)
    push_tissue_test
    ;;
  validate-vae-test)
    validate_vae_test "${2:-}"
    ;;
  publish-vae-test-inputs)
    publish_vae_test_inputs
    ;;
  status-vae-test-inputs)
    status_vae_test_inputs
    ;;
  verify-vae-test-inputs)
    verify_vae_test_inputs
    ;;
  push-vae-test)
    push_vae_test
    ;;
  validate-mil-test)
    validate_mil_test "${2:-}"
    ;;
  publish-mil-test-inputs)
    publish_mil_test_inputs
    ;;
  verify-mil-test-inputs)
    verify_mil_test_inputs
    ;;
  status-mil-test-inputs)
    status_mil_test_inputs
    ;;
  validate-mil-training)
    validate_mil_training "${2:-}"
    ;;
  build-mil-training-resume)
    build_mil_training_resume "${2:-}" "${3:-}" "${4:-}"
    ;;
  validate-mil-training-resume)
    validate_mil_training_resume "${2:-}" "${3:-}"
    ;;
  publish-mil-training-resume)
    publish_mil_training_resume "${2:-}"
    ;;
  verify-mil-training-resume)
    verify_mil_training_resume "${2:-}"
    ;;
  preflight-mil-training)
    preflight_mil_training "${2:-}"
    ;;
  publish-mil-training-inputs)
    publish_mil_training_inputs
    ;;
  verify-mil-training-inputs)
    verify_mil_training_inputs
    ;;
  build-supervised-calibration-input)
    build_supervised_calibration_input "${2:-}" "${3:-}"
    ;;
  publish-supervised-calibration-input)
    publish_supervised_calibration_input "${2:-}" "${3:-}"
    ;;
  verify-supervised-calibration-input)
    verify_supervised_calibration_input "${2:-}" "${3:-}"
    ;;
  build-supervised-calibration)
    build_supervised_calibration "${2:-}" "${3:-}" "${4:-}" "${5:-}"
    ;;
  preflight-supervised-calibration)
    preflight_supervised_calibration "${2:-}" "${3:-}" "${4:-}" "${5:-}"
    ;;
  build-wsi45630-completion)
    build_wsi45630_package
    ;;
  build-full-foreground-completion)
    full_foreground_package build
    ;;
  preflight-full-foreground-completion)
    full_foreground_package validate
    ;;
  publish-full-foreground-inputs)
    publish_full_foreground_inputs
    ;;
  verify-full-foreground-inputs)
    verify_full_foreground_inputs
    ;;
  output-full-foreground)
    output_full_foreground "${2:-}"
    ;;
  publish-wsi45630-completion-inputs)
    publish_wsi45630_inputs
    ;;
  verify-wsi45630-completion-inputs)
    verify_wsi45630_inputs
    ;;
  publish-full-wsi-capacity-inputs)
    publish_full_wsi_capacity_inputs
    ;;
  verify-full-wsi-capacity-inputs)
    verify_full_wsi_capacity_inputs
    ;;
  publish-cancer-topup-inputs)
    publish_cancer_topup_inputs
    ;;
  verify-cancer-topup-inputs)
    verify_cancer_topup_inputs
    ;;
  publish-latent-inputs)
    publish_latent_inputs
    ;;
  verify-latent-inputs)
    verify_latent_inputs
    ;;
  build-latent-resume)
    build_latent_resume "${2:-}" "${3:-}"
    ;;
  publish-latent-resume)
    publish_latent_resume "${2:-}"
    ;;
  verify-latent-resume)
    verify_latent_resume "${2:-}"
    ;;
  identity)
    kaggle_authenticated_username
    ;;
  push)
    # Only treat the first token as kernel_dir when it is a real path, not an option
    # flag, so `push --wait ...` still falls back to the default kernel_dir instead of
    # swallowing `--wait` as the directory.
    if [[ -n "${2:-}" && "$2" != --* ]]; then
      kernel_dir="$2"
      shift 2
    else
      kernel_dir="$default_kernel_dir"
      shift 1
    fi
    # --wait blocks after a successful push until the kernel settles, so a caller
    # can push-and-be-woken in a single backgrounded command. --wait-interval /
    # --wait-max / --wait-queued tune the RUNNING cadence, the RUNNING backstop,
    # and the QUEUED budget; all wait flags are consumed here and never forwarded
    # to the Kaggle CLI. Everything else passes through unchanged.
    push_wait=0
    push_wait_interval=300
    push_wait_max=180
    push_wait_queued=300
    push_passthrough=()
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
      --wait)
        push_wait=1
        ;;
      --wait-interval)
        if [[ "$#" -lt 2 ]]; then
          echo "error: --wait-interval requires a value" >&2
          exit 1
        fi
        push_wait_interval="$2"
        shift
        ;;
      --wait-max)
        if [[ "$#" -lt 2 ]]; then
          echo "error: --wait-max requires a value" >&2
          exit 1
        fi
        push_wait_max="$2"
        shift
        ;;
      --wait-queued)
        if [[ "$#" -lt 2 ]]; then
          echo "error: --wait-queued requires a value" >&2
          exit 1
        fi
        push_wait_queued="$2"
        shift
        ;;
      *)
        push_passthrough+=("$1")
        ;;
      esac
      shift
    done
    if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" ]]; then
      echo "error: set KAGGLE_PUSH_CONFIRMED=1 after explicit user permission" >&2
      exit 1
    fi
    if [[ "$push_wait" == "1" ]]; then
      require_remote_confirmed
    fi
    validate_kernel_dir "$kernel_dir"
    guard_push_ready "$kernel_dir"
    require_kaggle_sources_confirmed "$(metadata_path "$kernel_dir")"
    local_attention_probe_push=0
    local_attention_repair_probe_push=0
    local_global_capacity_push=0
    corrected_rotation_geometry_push=0
    decoded_transform_push=0
    functional_geometry_preflight_push=0
    functional_geometry_preflight_resume_push=0
    jvp_epsilon_calibration_push=0
    push_kernel_id="$(kernel_id_from_metadata "$kernel_dir")"
    if [[ "$kernel_dir" == "$corrected_rotation_geometry_kernel_dir" \
      || "$push_kernel_id" == "$corrected_rotation_geometry_kernel_id" ]]; then
      corrected_rotation_geometry_push=1
      if [[ "$push_wait" == "1" || "${#push_passthrough[@]}" -ne 0 ]]; then
        echo "error: Spec 0050 push forbids wait and all CLI overrides" >&2
        exit 1
      fi
    elif [[ "$kernel_dir" == "$decoded_transform_kernel_dir" \
      || "$push_kernel_id" == "$decoded_transform_kernel_id" ]]; then
      decoded_transform_push=1
      if [[ "$push_wait" == "1" || "${#push_passthrough[@]}" -ne 0 ]]; then
        echo "error: Spec 0051 push forbids wait and all CLI overrides" >&2
        exit 1
      fi
    elif [[ "$kernel_dir" == "$functional_geometry_preflight_kernel_dir" \
      || "$push_kernel_id" == "$functional_geometry_preflight_kernel_id" ]]; then
      functional_geometry_preflight_push=1
      if [[ "$push_wait" == "1" || "${#push_passthrough[@]}" -ne 0 ]]; then
        echo "error: Spec 0057 JVP ladder push forbids wait and all CLI overrides" >&2
        exit 1
      fi
    elif [[ "$kernel_dir" == "$functional_geometry_preflight_resume_kernel_dir" \
      || "$push_kernel_id" == "$functional_geometry_preflight_resume_kernel_id" ]]; then
      functional_geometry_preflight_resume_push=1
      if [[ "$push_wait" == "1" || "${#push_passthrough[@]}" -ne 0 ]]; then
        echo "error: Spec 0057 JVP ladder resume push forbids wait and all CLI overrides" >&2
        exit 1
      fi
    elif [[ "$kernel_dir" == "$jvp_epsilon_calibration_kernel_dir" \
      || "$push_kernel_id" == "$jvp_epsilon_calibration_kernel_id" ]]; then
      jvp_epsilon_calibration_push=1
      if [[ "$push_wait" == "1" || "${#push_passthrough[@]}" -ne 0 ]]; then
        echo "error: Spec 0058 calibration push forbids wait and all CLI overrides" >&2
        exit 1
      fi
    elif [[ "$kernel_dir" == "$local_global_capacity_kernel_dir" \
      || "$push_kernel_id" == "$local_global_capacity_kernel_id" ]]; then
      local_global_capacity_push=1
      if [[ "$push_wait" == "1" || "${#push_passthrough[@]}" -ne 0 ]]; then
        echo "error: Spec 0030 push forbids wait and all CLI overrides" >&2
        exit 1
      fi
    elif [[ "$kernel_dir" == "$local_attention_probe_kernel_dir" \
      || "$push_kernel_id" == "$local_attention_probe_kernel_id" ]]; then
      local_attention_probe_push=1
      local_attention_repair_probe_push=1
      if [[ "$push_wait" == "1" || "${#push_passthrough[@]}" -ne 0 ]]; then
        echo "error: Spec 0028 push forbids wait and all CLI overrides" >&2
        exit 1
      fi
    elif grep -q 'KAGGLE_LOCAL_ATTENTION_PROBE_READY = True' \
      "$kernel_dir/$(json_field "$(metadata_path "$kernel_dir")" code_file)"; then
      local_attention_probe_push=1
      if [[ "$push_wait" == "1" || "${#push_passthrough[@]}" -ne 0 ]]; then
        echo "error: Spec 0027 push forbids wait and all CLI overrides" >&2
        exit 1
      fi
    fi
    require_kaggle_cli
    if [[ "$local_attention_probe_push" == "1" ]]; then
      upload_kernel_dir="$kernel_dir"
      if [[ "$local_attention_repair_probe_push" == "1" ]]; then
        if ! upload_kernel_dir="$(
          make_local_attention_repair_probe_snapshot "$kernel_dir"
        )"; then
          echo "error: failed to create exact Spec 0028 upload snapshot" >&2
          exit 1
        fi
        claim_local_attention_repair_probe_push "$upload_kernel_dir"
      fi
      if ! push_response="$(kaggle_api kernels push -p "$upload_kernel_dir" 2>&1)"; then
        printf '%s\n' "$push_response" >&2
        exit 1
      fi
      printf '%s\n' "$push_response"
      if [[ "$push_response" =~ [Kk]ernel[[:space:]]version[[:space:]]([0-9]+)[[:space:]]successfully[[:space:]]pushed ]]; then
        accepted_version="${BASH_REMATCH[1]}"
      elif [[ "$push_response" =~ [Ss]uccessfully[[:space:]]pushed[[:space:]]version[[:space:]]([0-9]+) ]]; then
        accepted_version="${BASH_REMATCH[1]}"
      else
        echo "error: Kaggle did not confirm an accepted local-attention probe version" >&2
        exit 1
      fi
      if [[ "$local_attention_repair_probe_push" == "1" ]]; then
        finalize_local_attention_repair_probe_push \
          "$upload_kernel_dir" "$accepted_version"
      else
        record_local_attention_probe_push "$kernel_dir" "$accepted_version"
      fi
    else
      if [[ "${#push_passthrough[@]}" -ne 0 ]]; then
        echo "error: account-portable push forbids Kaggle CLI overrides" >&2
        echo "       encode runtime settings in guarded kernel metadata" >&2
        exit 1
      fi
      actor="$(kaggle_authenticated_username)"
      if [[ "$local_global_capacity_push" == "1" \
        && "$actor" != "maximshtefan" ]]; then
        echo "error: Spec 0030 corrective retry requires actor maximshtefan" >&2
        exit 1
      fi
      if [[ "$corrected_rotation_geometry_push" == "1" \
        || "$decoded_transform_push" == "1" \
        || "$functional_geometry_preflight_push" == "1" \
        || "$functional_geometry_preflight_resume_push" == "1" \
        || "$jvp_epsilon_calibration_push" == "1" ]]; then
        upload_kernel_dir="$(
          make_corrected_rotation_geometry_snapshot "$kernel_dir" "$actor"
        )"
      else
        upload_kernel_dir="$(
          make_account_portable_kernel_snapshot "$kernel_dir" "$actor"
        )"
      fi
      push_kernel_id="$(kernel_id_from_metadata "$upload_kernel_dir")"
      if [[ "$local_global_capacity_push" == "1" ]]; then
        claim_local_global_capacity_push "$kernel_dir" "$upload_kernel_dir" "$actor"
      fi
      if [[ "$corrected_rotation_geometry_push" == "1" ]]; then
        claim_corrected_rotation_geometry_push "$kernel_dir" "$upload_kernel_dir" "$actor"
      fi
      if [[ "$decoded_transform_push" == "1" ]]; then
        claim_decoded_transform_push "$kernel_dir" "$upload_kernel_dir" "$actor"
      fi
      if [[ "$functional_geometry_preflight_push" == "1" ]]; then
        preflight_functional_geometry_slug "$functional_geometry_preflight_kernel_id"
        claim_functional_geometry_preflight_push \
          "$kernel_dir" "$upload_kernel_dir" "$actor"
      fi
      if [[ "$functional_geometry_preflight_resume_push" == "1" ]]; then
        preflight_functional_geometry_slug "$functional_geometry_preflight_resume_kernel_id"
        claim_functional_geometry_preflight_resume_push \
          "$kernel_dir" "$upload_kernel_dir" "$actor"
      fi
      if [[ "$jvp_epsilon_calibration_push" == "1" ]]; then
        preflight_functional_geometry_slug "$jvp_epsilon_calibration_kernel_id"
        claim_jvp_epsilon_calibration_push \
          "$kernel_dir" "$upload_kernel_dir" "$actor"
      fi
      if ! push_response="$(
        kaggle_api kernels push -p "$upload_kernel_dir" 2>&1
      )"; then
        printf '%s\n' "$push_response" >&2
        exit 1
      fi
      printf '%s\n' "$push_response"
      if ! canonical_kernel_reference="$(
        confirmed_kernel_reference "$push_response"
      )"; then
        echo "error: Kaggle did not explicitly confirm a canonical kernel URL and version" >&2
        exit 1
      fi
      if [[ "$functional_geometry_preflight_push" == "1" \
        && "$canonical_kernel_reference" \
          != "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5/1" ]]; then
        echo "error: Spec 0057 JVP ladder parent push accepted an unexpected canonical reference" >&2
        exit 1
      fi
      if [[ "$functional_geometry_preflight_resume_push" == "1" \
        && "$canonical_kernel_reference" \
          != "maximshtefan/eqvae-functional-geometry-preflight-04a08ab5-resume/1" ]]; then
        echo "error: Spec 0057 JVP ladder resume push accepted an unexpected canonical reference" >&2
        exit 1
      fi
      if [[ "$jvp_epsilon_calibration_push" == "1" \
        && "$canonical_kernel_reference" \
          != "maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5/1" ]]; then
        echo "error: Spec 0058 calibration push accepted an unexpected canonical reference" >&2
        exit 1
      fi
      push_kernel_id="$canonical_kernel_reference"
      launch_receipt="$(
        record_account_portable_launch \
          "$kernel_dir" "$upload_kernel_dir" "$canonical_kernel_reference"
      )"
      echo "ok: canonical Kaggle launch saved at $launch_receipt"
    fi
    if [[ "$push_wait" == "1" ]]; then
      echo "push: waiting for ${push_kernel_id} to settle..."
      wait_kernel_until_settled \
        "$push_kernel_id" "$push_wait_interval" "$push_wait_max" \
        "$push_wait_queued"
    fi
    ;;
  status)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-launch)
    kernel_id="$(kernel_reference_from_launch_receipt "${2:?launch receipt required}")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-setup)
    kernel_id="$(kernel_id_from_metadata "$setup_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-real-data-runtime-pretest)
    kernel_id="$(kernel_id_from_metadata "$real_data_runtime_pretest_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-runtime-selection)
    kernel_id="$(kernel_id_from_metadata "$runtime_selection_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-selected-runtime-debug)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_debug_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-selected-runtime-lr-range)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_lr_range_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_full_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-fixed25-selector)
    kernel_id="$(kernel_id_from_metadata "$fixed25_selector_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-so2-architecture-probe)
    kernel_id="$(kernel_id_from_metadata "$so2_architecture_probe_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-so2-runtime-readiness)
    kernel_id="$(kernel_id_from_metadata "$so2_runtime_readiness_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-so2-prelaunch)
    kernel_id="$(kernel_id_from_metadata "$so2_prelaunch_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-so2-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$so2_full_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  wait)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${3:-300}" "${4:-180}" "${5:-300}"
    ;;
  wait-fixed25-selector)
    kernel_id="$(kernel_id_from_metadata "$fixed25_selector_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${2:-300}" "${3:-180}" "${4:-300}"
    ;;
  wait-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_full_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${2:-300}" "${3:-480}" "${4:-600}"
    ;;
  wait-so2-architecture-probe)
    kernel_id="$(kernel_id_from_metadata "$so2_architecture_probe_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${2:-300}" "${3:-36}" "${4:-600}"
    ;;
  wait-so2-runtime-readiness)
    kernel_id="$(kernel_id_from_metadata "$so2_runtime_readiness_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${2:-300}" "${3:-36}" "${4:-600}"
    ;;
  output)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    output_dir="${3:-$default_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-launch)
    kernel_id="$(kernel_reference_from_launch_receipt "${2:?launch receipt required}")"
    output_dir="${3:?output directory required}"
    require_remote_confirmed
    require_kaggle_cli
    if [[ -e "$output_dir" ]]; then
      echo "error: output-launch requires a new output directory: $output_dir" >&2
      exit 1
    fi
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    record_kaggle_download \
      kernel "$kernel_id" "$output_dir" kaggle_output_receipt.json
    ;;
  output-mil-training)
    output_mil_training "${2:-}" "${3:-}"
    ;;
  output-mil-test)
    output_mil_test "${2:-}" "${3:-}"
    ;;
  score-mil-test)
    score_mil_test "${2:-}" "${3:-}" "${4:-}"
    ;;
  output-tissue-test)
    output_tissue_test "${2:-}" "${3:-}"
    ;;
  score-tissue-test)
    score_tissue_test "${2:-}" "${3:-}" "${4:-}"
    ;;
  output-vae-test)
    output_vae_test "${2:-}" "${3:-}"
    ;;
  score-vae-test)
    score_vae_test "${2:-}" "${3:-}" "${4:-}"
    ;;
  resume-score-vae-test)
    [[ "$#" -eq 1 ]] || {
      echo "error: resume-score-vae-test accepts no path overrides" >&2
      exit 1
    }
    resume_score_vae_test
    ;;
  dataset-download)
    dataset_reference="$(
      validated_versioned_reference "${2:?owner/dataset/version required}"
    )"
    output_dir="${3:?output directory required}"
    require_remote_confirmed
    require_kaggle_cli
    if [[ -e "$output_dir" ]]; then
      echo "error: dataset-download requires a new output directory: $output_dir" >&2
      exit 1
    fi
    mkdir -p "$output_dir"
    kaggle_api datasets download "$dataset_reference" -p "$output_dir" --unzip
    record_kaggle_download \
      dataset "$dataset_reference" "$output_dir" kaggle_dataset_receipt.json
    ;;
  output-local-attention-probe)
    require_remote_confirmed
    require_kaggle_cli
    if [[ ! -f "$local_attention_probe_push_receipt" ]]; then
      echo "error: Spec 0027 output requires the consumed push receipt" >&2
      exit 1
    fi
    if [[ -e "$local_attention_probe_output_dir" ]]; then
      echo "error: refusing to mix or overwrite Spec 0027 evidence" >&2
      exit 1
    fi
    accepted_version="$(python3 - "$local_attention_probe_push_receipt" <<'PYLOCALATTENTIONVERSION'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
version = receipt.get("accepted_version")
if (
    receipt.get("schema_version") != "spec0027.push_receipt.v1"
    or receipt.get("authority_consumed") is not True
    or receipt.get("kernel_id")
    != "maximusshtefan/eqvae-wsi45630-local-attention-probe"
    or isinstance(version, bool)
    or not isinstance(version, int)
    or version < 1
):
    raise SystemExit("invalid Spec 0027 push receipt")
print(version)
PYLOCALATTENTIONVERSION
)"
    versioned_kernel_id="$local_attention_probe_kernel_id/$accepted_version"
    stage_dir="$(mktemp -d "$TMPDIR/spec0027_retrieval.XXXXXX")"
    kaggle_api kernels output "$versioned_kernel_id" -p "$stage_dir" \
      --file-pattern '^spec0027_local_attention_probe[.]json$'
    kaggle_api kernels logs "$versioned_kernel_id" >"$stage_dir/kaggle.log"
    if [[ ! -f "$stage_dir/spec0027_local_attention_probe.json" \
      || ! -s "$stage_dir/kaggle.log" ]]; then
      echo "error: exact Spec 0027 JSON and nonempty Kaggle log are required" >&2
      exit 1
    fi
    python3 - \
      "$stage_dir/spec0027_local_attention_probe.json" \
      "$stage_dir/kaggle.log" \
      "$stage_dir/retrieval_receipt.json" "$accepted_version" <<'PYLOCALATTENTIONOUTPUT'
import hashlib
import json
import sys
from pathlib import Path

artifact, log, receipt = (Path(value) for value in sys.argv[1:4])
if not artifact.is_file() or not log.is_file():
    raise SystemExit("Spec 0027 JSON and Kaggle log are both required")
payload = json.loads(artifact.read_text(encoding="utf-8"))
if payload.get("spec") != "0027":
    raise SystemExit("retrieved artifact is not Spec 0027 evidence")
record = {
    "schema_version": "spec0027.retrieval_receipt.v1",
    "kernel_id": "maximusshtefan/eqvae-wsi45630-local-attention-probe",
    "accepted_version": int(sys.argv[4]),
    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
}
receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PYLOCALATTENTIONOUTPUT
    mkdir -p "$(dirname "$local_attention_probe_output_dir")"
    mv "$stage_dir" "$local_attention_probe_output_dir"
    ;;
  output-local-attention-repair-probe)
    require_remote_confirmed
    require_kaggle_cli
    if [[ ! -f "$local_attention_repair_probe_push_receipt" ]]; then
      echo "error: Spec 0028 output requires the consumed push receipt" >&2
      exit 1
    fi
    if [[ -e "$local_attention_repair_probe_output_dir" ]]; then
      echo "error: refusing to mix or overwrite Spec 0028 evidence" >&2
      exit 1
    fi
    accepted_version="$(python3 - "$local_attention_repair_probe_push_receipt" <<'PYLOCALATTENTIONREPAIRVERSION'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    receipt.get("schema_version") != "spec0028.push_receipt.v1"
    or receipt.get("authority_consumed") is not True
    or receipt.get("kernel_id")
    != "maximusshtefan/eqvae-wsi45630-local-attention-probe"
    or receipt.get("accepted_version") != 2
):
    raise SystemExit("invalid Spec 0028 push receipt")
print(2)
PYLOCALATTENTIONREPAIRVERSION
)"
    versioned_kernel_id="$local_attention_repair_probe_kernel_id/$accepted_version"
    mkdir -p "$(dirname "$local_attention_repair_probe_output_dir")"
    stage_dir="$(mktemp -d "${local_attention_repair_probe_output_dir}.staging.XXXXXX")"
    kaggle_api kernels output "$versioned_kernel_id" -p "$stage_dir" \
      --file-pattern '^spec0028_local_attention_repair_probe[.]json$'
    kaggle_api kernels logs "$versioned_kernel_id" >"$stage_dir/kaggle.log"
    if [[ ! -f "$stage_dir/spec0028_local_attention_repair_probe.json" \
      || ! -s "$stage_dir/kaggle.log" ]]; then
      echo "error: exact Spec 0028 JSON and nonempty Kaggle log are required" >&2
      exit 1
    fi
    python3 - \
      "$stage_dir/spec0028_local_attention_repair_probe.json" \
      "$stage_dir/kaggle.log" \
      "$stage_dir/retrieval_receipt.json" <<'PYLOCALATTENTIONREPAIROUTPUT'
import hashlib
import json
import sys
from pathlib import Path

artifact, log, receipt = (Path(value) for value in sys.argv[1:])
payload = json.loads(artifact.read_text(encoding="utf-8"))
if payload.get("spec") != "0028":
    raise SystemExit("retrieved artifact is not Spec 0028 evidence")
record = {
    "schema_version": "spec0028.retrieval_receipt.v1",
    "requested_kernel_id": "maximusshtefan/eqvae-wsi45630-local-attention-probe",
    "kernel_id": "maximusshtefan/eqvae-wsi45630-local-attention-repair-probe",
    "accepted_version": 2,
    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
}
receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PYLOCALATTENTIONREPAIROUTPUT
    mv "$stage_dir" "$local_attention_repair_probe_output_dir"
    ;;
  output-real-data-runtime-pretest)
    kernel_id="$(kernel_id_from_metadata "$real_data_runtime_pretest_kernel_dir")"
    output_dir="${2:-$real_data_runtime_pretest_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-runtime-selection)
    kernel_id="$(kernel_id_from_metadata "$runtime_selection_kernel_dir")"
    output_dir="${2:-$runtime_selection_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-selected-runtime-debug)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_debug_kernel_dir")"
    output_dir="${2:-$selected_runtime_debug_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-selected-runtime-lr-range)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_lr_range_kernel_dir")"
    output_dir="${2:-$selected_runtime_lr_range_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_full_kernel_dir")"
    output_dir="${2:-$selected_runtime_full_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-fixed25-selector)
    kernel_id="$(kernel_id_from_metadata "$fixed25_selector_kernel_dir")"
    output_dir="${2:-$fixed25_selector_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-so2-architecture-probe)
    kernel_id="$(kernel_id_from_metadata "$so2_architecture_probe_kernel_dir")"
    output_dir="${2:-$so2_architecture_probe_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-so2-runtime-readiness)
    kernel_id="$(kernel_id_from_metadata "$so2_runtime_readiness_kernel_dir")"
    output_dir="${2:-$so2_runtime_readiness_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-so2-prelaunch)
    kernel_id="$(kernel_id_from_metadata "$so2_prelaunch_kernel_dir")"
    output_dir="${2:-$so2_prelaunch_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-so2-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$so2_full_kernel_dir")"
    output_dir="${2:-$so2_full_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-setup)
    kernel_id="$(kernel_id_from_metadata "$setup_kernel_dir")"
    output_dir="${2:-$setup_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  pull)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    kernel_dir="${3:-$default_kernel_dir}"
    require_remote_confirmed
    if [[ "${KAGGLE_PULL_CONFIRMED:-}" != "1" ]]; then
      echo "error: set KAGGLE_PULL_CONFIRMED=1 after explicit user permission" >&2
      exit 1
    fi
    guard_clean_kernel_dir "$kernel_dir"
    require_kaggle_cli
    kaggle_api kernels pull "$kernel_id" -p "$kernel_dir"
    ;;
  pull-launch)
    kernel_id="$(kernel_reference_from_launch_receipt "${2:?launch receipt required}")"
    kernel_dir="${3:?kernel directory required}"
    require_remote_confirmed
    if [[ "${KAGGLE_PULL_CONFIRMED:-}" != "1" ]]; then
      echo "error: set KAGGLE_PULL_CONFIRMED=1 after explicit user permission" >&2
      exit 1
    fi
    if [[ -e "$kernel_dir" ]]; then
      echo "error: pull-launch requires a new kernel directory: $kernel_dir" >&2
      exit 1
    fi
    require_kaggle_cli
    kaggle_api kernels pull "$kernel_id" -p "$kernel_dir"
    record_kaggle_download \
      kernel "$kernel_id" "$kernel_dir" kaggle_kernel_pull_receipt.json
    ;;
  *)
    usage
    exit 1
    ;;
esac
