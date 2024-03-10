import os

from .cache_engine import CacheConfig

from .cache_engine import CacheEngine as VllmCacheEngine
from .hip_cache_engine import HipCacheEngine
from .map_cache_engine import MapCacheEngine

cache_engine_method = os.getenv('CACHE_ENGINE', 'vllm')

if cache_engine_method == 'vllm':
    CacheEngine = VllmCacheEngine
elif cache_engine_method == 'map':
    CacheEngine = MapCacheEngine
elif cache_engine_method == 'hip':
    CacheEngine = HipCacheEngine
else:
    raise Exception()