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

for dirname in listdir(source_path):
    if 'A06' in dirname:
        print("Processing directory ", dirname)

        imgs = [f for f in listdir(join(source_path, dirname, suffix))]
        
        # load target img: use the first image in every directory as the first image
        target_img_path = join(source_path, dirname.replace('A', 'H', 1), suffix, imgs[0].replace('A', 'H', 1))
        target_img = stain_utils.read_image(target_img_path)
        print("Target image selected as: ", target_img_path)

        # normalizers
        normalizer_rd = stainNorm_Reinhard.Normalizer()
        normalizer_rd.fit(target_img)

        normalizer_mk = stainNorm_Macenko.Normalizer()
        normalizer_mk.fit(target_img)

        normalizer_vd = vahadane.vahadane()
        Wt, Ht = normalizer_vd.stain_separate(target_img)

        print("Finished preparing source image...")

        for img in imgs[1:]:
            I = stain_utils.read_image(join(source_path, dirname, suffix, img))

            print("Processing image: ", img)

            # reinhard
            out = normalizer_rd.transform(I)
            cv.imwrite(join(save_path, "reinhard", img), out)

            # macenko
            out = normalizer_mk.transform(I)
            cv.imwrite(join(save_path, "macenko", img), out)

            # vahadane
            Ws, Hs = normalizer_vd.stain_separate(I)
            out = normalizer_vd.SPCN(I, Ws, Hs, Wt, Ht)
            cv.imwrite(join(save_path, "vahadane", img), out)
