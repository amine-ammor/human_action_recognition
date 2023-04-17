# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### caching of the dataset computation along with some interesting metrics

# + tags=["parameters"]
# parameters

truncate = 30
clear_cache = True
location = '/root/.cache/keypoints_test'

# + tags=["notebook_call"]
from ploomber_engine.ipython import PloomberClient

path_training_setup = "../training_setup.ipynb"
client = PloomberClient.from_path(path_training_setup,
                                  cwd="../",
                                  remove_tagged_cells=["parameters","notebook_call","dataloader"])
namespace_client = client.get_namespace(dict(truncate=truncate,clear_cache=clear_cache,location=location))
for el in ["video_kpt_dataset","truncate"]:
    globals()[el] = namespace_client[el]

# +
import os

def get_size_directory(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

print(get_size_directory(location), 'bytes')
# -

res = []
for _ in range(2):
    import time
    start= time.time()
    video_kpt_dataset.with_frame_padding = False
    res.append([el for el in video_kpt_dataset])
    stop = time.time()
    stop-start
    print(stop-start)

# sanity check tests
res_before_cache,res_after_cache = res[0],res[1]
import numpy as np
for res0,res1 in zip(res_before_cache,res_after_cache):
    for key in ['kpts', 'is_detection_present']:
        assert np.all(res0[key] == res1[key])
    

#memory after caching the keypoints
print(get_size_directory(location), 'bytes')

res = video_kpt_dataset.number_of_frames,video_kpt_dataset.max_number_of_frames

for _ in range(2):
    import time
    start= time.time()
    video_kpt_dataset.with_frame_padding = True
    [el for el in video_kpt_dataset]
    stop = time.time()
    print(stop-start)

#memory after caching the keypoints along with padding
print(get_size_directory(location), 'bytes')
