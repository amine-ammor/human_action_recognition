"""
    python utilities to download data from websites interactively
"""
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from pathlib import Path
from tqdm import tqdm
import requests,zipfile,os,shutil

class RemoteDataFetcher:
    def __init__(self,website_url,expression,folder_dst) -> None:
        """
            website_url : url where to download the content
            folder_dst : folder where to store the downloaded data
        """
        self.website_url = website_url
        self.folder_dst = Path(folder_dst)

        self.folder_dst.mkdir(parents=True,exist_ok=True)

        self.urls = self._get_urls(expression)
        
        self.urls_to_zip_loc = {url:self.folder_dst.joinpath(Path(url).name) for url in self.urls}

        self.zip_lod_to_folder_loc = {path:Path(str(path).replace(".zip","")) 
                         for path in self.urls_to_zip_loc.values() }
        
    def _get_urls(self,expression=".zip"):
        """
            get from the website url all the urls with specific expression
            within the string
        """
        html_page = urlopen(Request(self.website_url))
        soup = BeautifulSoup(html_page)
        urls = [el.attrs["href"] for el in soup.find_all("a",href=True)]
        urls = [url for url in urls if expression in url]
        return urls
    
    def download(self):
        
        for url,path in tqdm(self.urls_to_zip_loc.items()):
            if not(path.exists()) and not(self.zip_lod_to_folder_loc[path].exists()):
                print(url)
                open(str(path),"wb").write(requests.get(url).content)

    def unzip(self):
        for path_zip,folder in tqdm(self.zip_lod_to_folder_loc.items()):
            if path_zip.exists():
                try:
                    print(path_zip)
                    with zipfile.ZipFile(str(path_zip), 'r') as zip_ref:
                        zip_ref.extractall(str(folder))
                except zipfile.BadZipFile:
                    print(f"deleting zipfile {path_zip} as it is damaged (try to download it again)")
                    os.remove(path_zip)
    def clean(self):
        for path_zip,folder in tqdm(self.zip_lod_to_folder_loc.items()):
            if path_zip.exists():
                # we check onyl by file name to check if two files are identical (under the archive
                # and the extracted which is not sufficient, but gives a nice rapid checkup for a first version)
                name_tgt = [file.filename for file in zipfile.ZipFile(str(path_zip), 'r').filelist]
                names = [path.name for path in list(Path(self.zip_lod_to_folder_loc[path_zip]).rglob("*"))]
                if set(name_tgt) == set(names):
                    print("here")
                    import os
                    os.remove(path_zip)