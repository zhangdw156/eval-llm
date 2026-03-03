import os
from evalscope import TaskConfig, run_task
import dotenv

# 从脚本所在目录加载 .env-minimax（变量: model, api_url, api_key）
# 对应 vllm configs/Qwen3.5-0.8B.yaml（port 8000）
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '.env-minimax'))

_model = os.getenv('model')
_api_url = os.getenv('api_url')
_api_key = os.getenv('api_key')
if not _model or not _api_url or not _api_key:
    raise SystemExit(
        '缺少环境变量: 请在 .env-minimax 中设置 model, api_url, api_key'
    )

task_cfg = TaskConfig(
    model=_model,
    api_url=_api_url,
    api_key=_api_key,
    eval_type='openai_api',

    datasets=['bfcl_v4'],
    dataset_args={
        'bfcl_v4': {
            # 评测子任务列表
            'subset_list': [
                'multi_turn_base',
                'multi_turn_miss_func',
                'multi_turn_miss_param',
                'multi_turn_long_context',
            ],
            'extra_params':{
                # 模型在函数名称中拒绝使用点号（`.`）；设置此项，以便在评估期间自动将点号转换为下划线。
                'underscore_to_dot': True,
                # 模式是否为函数调用模型（Function Calling Model），如果是则会启用函数调用相关的配置；否则会使用prompt绕过函数调用。
                'is_fc_model': True,
            }
        }
    },
    generation_config={
        'temperature': 0, # 只支持设置温度参数，其他参数会被忽略
    },
    use_cache='outputs/bfcl_v4', # 建议设置缓存目录，评测出错时可以加快重跑速度
    # repeats=4, # k for pass@k
    eval_batch_size=16,
    # limit=5,  ## for quick test
)

run_task(task_cfg)
