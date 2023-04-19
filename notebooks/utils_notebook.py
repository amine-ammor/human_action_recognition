"""
    wrapper around ploomber-engine
"""

# fill the cache and print informations about caching in other notebook
from pathlib import Path
from ploomber_engine.ipython import PloomberClient
import psutil


def execute_notebook(path_notebook,remove_tagged_cells=None,**params):
    """
        ignoring cells wiht specfic taggs and return namespace
    """
    path_notebook = Path(path_notebook)
    assert path_notebook.exists()        
        
    client = PloomberClient.from_path(path_notebook,remove_tagged_cells=remove_tagged_cells)
    namespace = client.get_namespace(params)
    del client
    return namespace


def print_memory():
    import torch
    if torch.cuda.is_available():
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        print(f'free gpu memory     : {psutil._common.bytes2human(info.free)}')
        print(f'allocated gpu memory     : {psutil._common.bytes2human(info.used)}')

    print(f"free ram {psutil._common.bytes2human(psutil.virtual_memory().free)}")
    print(f"used ram {psutil._common.bytes2human(psutil.virtual_memory().used)}")


import jupytext,time,os

class NotebookPyExecutor:
    """ execute notebook written as python file sa notebook using jupytext and ploomber-engine"""
    def __init__(self,path_training_setup,remove_tagged_cells=None,**params):
        self.path_training_setup = str(Path(path_training_setup).resolve())
        
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path_training_setup_as_notebook = Path(self.temp_dir.name).joinpath(Path(self.path_training_setup).name.replace(".py",".ipynb"))
        
        self.remove_tagged_cells = remove_tagged_cells
        self.params = params
        print_memory()
        
    def __enter__(self):
        jupytext.write(nb=jupytext.read(self.path_training_setup),
                       fp =self.path_training_setup_as_notebook)
        self.namespace = execute_notebook(self.path_training_setup_as_notebook,self.remove_tagged_cells,**self.params)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if os.path.exists(self.path_training_setup_as_notebook):
            os.remove(self.path_training_setup_as_notebook)
        del self.namespace
        self.temp_dir.cleanup()
        print_memory()

