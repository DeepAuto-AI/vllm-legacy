from .cache_engine import CacheConfig
from .cache_engine import CacheEngine as VllmCacheEngine
from .hip_cache_engine import HipCacheEngine
import os

CacheEngine = VllmCacheEngine
# if 'vllm' in [os.getenv('PAGED_ATTENTION_METHOD', 'timber'), os.getenv('PROMPT_ATTENTION_METHOD', 'timber')]:
#     CacheEngine = VllmCacheEngine
# else:
#     CacheEngine = HipCacheEngine