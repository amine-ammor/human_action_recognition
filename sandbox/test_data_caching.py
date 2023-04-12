from pathlib import Path


dir_file = Path(__file__).parent

path_notebook_for_caching = dir_file.joinpath("../notebooks/performance_tests/caching_tests_and_metrics.ipynb")

client = PloomberClient.from_path(path_notebook_for_caching,cwd=path_notebook_for_caching.parent)

notebook_node = client.execute(dict(truncate=30,clear_cache=True))
import nbformat
nbformat.write(notebook_node,"./sandbox/output_notebooks/cache_results.ipynb")