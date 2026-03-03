import os
from evalscope import TaskConfig, run_task
import dotenv

# 从脚本所在目录加载 .env-minimax（变量: model, api_url, api_key）
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

    datasets=['tau2_bench'],
    dataset_args={
        'tau2_bench': {
            'subset_list': ['airline', 'retail', 'telecom'], 
            'extra_params': {
                'user_model': _model,
                'api_key': _api_key,
                'api_base': _api_url,
                'generation_config': {
                    'temperature': 0.7,
                }
            }
        }
    },

    eval_batch_size=5,
    limit=5,
    generation_config={
        'temperature': 0.6,
    },
)

run_task(task_cfg)