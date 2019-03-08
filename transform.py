import stain_utils
import stainNorm_Macenko
import stainNorm_Reinhard
import vahadane

import cv2 as cv

from os import listdir, mkdir
from os.path import isfile, join, isdir

# paths
source_path = "/home/fred/Projects/srt-cancer-img/pair-data/"
suffix = "frames/x40/"
save_path = "/home/fred/Projects/srt-cancer-img/pair-data/output/"

# normalizers
normalizer_rd = stainNorm_Reinhard.Normalizer()
normalizer_mk = stainNorm_Macenko.Normalizer()
normalizer_vd = vahadane.vahadane()

for dirname in listdir(source_path):
    if 'A06' in dirname:
        print("Processing directory ", dirname)

        imgs = [f for f in listdir(join(source_path, dirname, suffix))]
        
        for img in imgs[1:]:
            source = stain_utils.read_image(join(source_path, dirname, suffix, img))
            target = stain_utils.read_image(join(source_path, dirname.replace('A', 'H', 1), suffix, img.replace('A', 'H', 1)))

            print("Processing image: ", img)

            # load and fit target img 
            normalizer_rd.fit(target)
            normalizer_mk.fit(target)
            Wt, Ht = normalizer_vd.stain_separate(target)

            # reinhard
            out = normalizer_rd.transform(source)
            out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
            cv.imwrite(join(save_path, "reinhard", img), out)

            # macenko
            out = normalizer_mk.transform(source)
            out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
            cv.imwrite(join(save_path, "macenko", img), out)

            # vahadane
            Ws, Hs = normalizer_vd.stain_separate(source)
            out = normalizer_vd.SPCN(source, Ws, Hs, Wt, Ht)
            out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
            cv.imwrite(join(save_path, "vahadane", img), out)
