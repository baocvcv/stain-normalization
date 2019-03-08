import tarfile

from os import listdir, mkdir
from os.path import isfile, join, isdir

paths = ["/home/fred/Projects/srt-cancer-img/readings&files/Data/testing_aperio/", "/home/fred/Projects/srt-cancer-img/readings&files/Data/testing_hamamatsu/"]
save_path = "/home/fred/Projects/srt-cancer-img/pair-data/"

for path in paths:
    for f in listdir(path):
        if isfile(join(path, f)):
            tar = tarfile.open(join(path,f), "r:gz")
            for member in tar.getmembers():
                if 'x40' in member.name:
                    tar.extract(member, path=save_path)
