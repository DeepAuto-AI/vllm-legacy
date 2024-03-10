import os
from functools import partial

from .cache_engine import CacheConfig
from .cache_engine import CacheEngine as VllmCacheEngine
from .hip_cache_engine import HipCacheEngine
from .map_cache_engine import MapCacheEngine, VMapCacheEngine, KVMapCacheEngine

cache_engine_method = os.getenv('CACHE_ENGINE', 'vllm')

if cache_engine_method == 'vllm':
    CacheEngine = VllmCacheEngine
elif cache_engine_method == 'map':
    CacheEngine = VMapCacheEngine
elif cache_engine_method == 'map_kv':
    CacheEngine = KVMapCacheEngine
elif cache_engine_method == 'hip':
    CacheEngine = HipCacheEngine
else:
    raise Exception()