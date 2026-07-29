#!/usr/bin/env bash
set -euo pipefail

job_prefix="${1:-x-navdp}"
embodiment="${2:-quadruped}"
scene="${3:-internscene_commercial}"
task="${4:-pointgoal}"
start_idx="${5:-0}"
end_idx="${6:-19}"
checkpoint="${7:-/cpfs/user/yangtianyu/NavDP/baselines/x-navdp/ckpt/x-navdp_more/navdp_rl_24000.ckpt}"

repo_root="/cpfs/user/yangtianyu/NavDP/baselines/x-navdp"
dlc_bin="/mnt/data/yangtianyu/dlc"
port=10093

if [[ "${task}" != "pointgoal" ]]; then
    echo "x-navdp eval currently supports task=pointgoal, got: ${task}" >&2
    exit 1
fi

case "${scene}" in
    internscene_home|internscenes_home|home)
        config_scene="internscene_home"
        ;;
    internscene_commercial|internscenes_commercial|commercial)
        config_scene="internscene_commercial"
        ;;
    clutter_easy|cluttered_easy)
        config_scene="clutter_easy"
        ;;
    clutter_hard|cluttered_hard)
        config_scene="clutter_hard"
        ;;
    *)
        echo "Unknown scene '${scene}'. Expected internscene_home, internscene_commercial, clutter_easy, or clutter_hard." >&2
        exit 1
        ;;
esac

config_file="eval/config/eval_${task}/${embodiment}_${config_scene}.yaml"
if [[ ! -f "${repo_root}/${config_file}" ]]; then
    echo "Config file not found: ${repo_root}/${config_file}" >&2
    exit 1
fi

for i in $(seq "${start_idx}" "${end_idx}"); do
    "${dlc_bin}" submit pytorchjob \
        --name="${job_prefix}_eval_${embodiment}_${scene}_${task}_${i}" \
        --command="bash -i -c 'set -e; \
        export PATH=/root/miniconda3/bin:\$PATH; \
        source /root/.bashrc; \
        unset http_proxy; unset https_proxy; \
        cd ${repo_root}; \
        conda activate navrl; \
        export PYTHONPATH=${repo_root}:\${PYTHONPATH:-}; \
        export ACADOS_SOURCE_DIR=/cpfs/user/yangtianyu/acados; \
        export LD_LIBRARY_PATH=/cpfs/user/yangtianyu/acados/lib:\${LD_LIBRARY_PATH:-}; \
        python -c \"import sys, isaaclab; print(sys.executable)\"; \
        bash eval/scripts/start_policy_server.sh --port ${port} --embodiment ${embodiment} --checkpoint ${checkpoint} --device cuda:0 & \
        server_pid=\$!; \
        trap \"kill \${server_pid} 2>/dev/null || true\" EXIT; \
        sleep 5; \
        bash eval/scripts/run_evaluation.sh --config_file ${config_file} --scene_index ${i} --server_port ${port} --device cuda:0; \
        eval_status=\$?; \
        kill \${server_pid} 2>/dev/null || true; \
        wait \${server_pid} 2>/dev/null || true; \
        exit \${eval_status}'" \
        --data_sources=d-8wz4emfs21s5ajs9oz:v1:/mnt/data/,d-d49o5g0h2818sw8j1g:v1:/shared/smartbot/,d-rvm7u26zzahla2yrh3:v1:/cpfs/user/yangtianyu/ \
        --resource_id=quota1r947pmazvk \
        --tags="CloneFromJobID=dlc10wd1c8vf2faj" \
        --workspace_id=270969 \
        --vpc_id=vpc-2zef1skt5zeyxqsntfobm \
        --switch_id=vsw-2zek6cpni6tjimpyt1c9m \
        --security_group_id=sg-2ze0lfnfmc34b3knkqet \
        --priority=7 \
        --default_route=eth1 \
        --job_max_running_time_minutes=43200 \
        --workers=1 \
        --worker_image=pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/yangtianyu:yty-navrl2 \
        --worker_cpu=14 \
        --worker_memory=100Gi \
        --worker_shared_memory=100Gi \
        --worker_gpu=1
done
