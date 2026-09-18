"""Bounded immutable image preprocessing; model state stays in the main process."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
from PIL import Image

class FrameLoader:
    def __init__(self,root,image_processor,workers=1,cache_size=32):
        self.root=Path(root);self.processor=image_processor
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.cache=OrderedDict();self.cache_size=cache_size
    def _load(self,frame):
        with Image.open(self.root/frame.image_path) as image:
            data=self.processor.preprocess([image.convert('RGB')],return_tensors='pt')
        return data['pixel_values'],data['image_grid_thw'][0]
    def prefetch(self,frames):
        for frame in frames:
            if frame.key not in self.cache:self.cache[frame.key]=self.pool.submit(self._load,frame)
    def get(self,frame):
        self.prefetch([frame]);future=self.cache[frame.key];self.cache.move_to_end(frame.key)
        value=future.result()
        while len(self.cache)>self.cache_size:
            key,old=self.cache.popitem(last=False)
            # Futures already running are allowed to finish; at most one update is prefetched.
            if not old.done():old.cancel()
        return value
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
