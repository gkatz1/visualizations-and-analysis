from detail import Detail
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import pylab
import json
import random


def get_rand_color_map():
    return matplotlib.colors.ListedColormap(np.random.rand(256,3))

CLASS_NAMES = [
               'background',
               'aeroplane',
               'bicycle',
               'bird',
               'boat',
               'bottle',
               'bus',
               'car',
               'cat',
               'chair',
               'cow',
               'diningtable',
               'dog',
               'horse',
               'motorbike',
               'person',
               'potted-plant',
               'sheep',
               'sofa',
               'train',
               'tvmonitor',
               'aeroplane_part',
               'bicycle_part',
               'bird_part',
               'boat_part',
               'bottle_part',
               'bus_part',
               'car_part',
               'cat_part',
               'chair_part',
               'cow_part',
               'diningtable_part',
               'dog_part',
               'horse_part',
               'motorbike_part',
               'person_part',
               'potted-plant',
               'sheep_part',
               'sofa_part',
               'train_part',
               'tvmonitor_part']


pylab.rcParams['figure.figsize'] = (10.0, 8.0)
ann_file = '../../json/trainval_merged.json'
im_dir = '../../VOCdevkit/VOC2010/JPEGImages'
pahse = 'trainval'

details = Detail(ann_file, im_dir, pahse)

# print "\n*** info ***"
# details.info()

# cats = details.getCats()

# print '%d objects categories (including animals)' % len(details.getCats(supercat='object'))
# print('%d background categories' % len(details.getCats(supercat='background')))
# print('%d categories total' % len(cats))
# names = set([cat['supercategory'] for cat in cats])
# print('\nSupercategories: \n\t', ' '.join(names))

# print('Categories: ')
# for cat in cats:
#     print('\t{name} ({category_id}): \t{supercategory}'.format(**cat))


# print('%d images in the current phase (%s)' % (len(details.getImgs()), details.phase))
# imgs = details.getImgs(cats=['person', 'motorbike']);
# print('%d images contain both persons and motorbikes' % len(imgs))

_, axarr = plt.subplots(1,2)


# img = imgs[random.randrange(len(imgs))]
img = '2008_007573'
pt = [261, 119]
cat_idx = 8
details.showImg(img, ax=axarr[0], wait=True)

# cats = details.getCats(imgs=img)
# print("")
# print(['%s (%d) : %s' % (cat['name'], cat['category_id'], cat['supercategory']) for cat in cats])

print "****"
print "{} --> ".format(cat_idx) + CLASS_NAMES[cat_idx]

# mask
# mask = details.getMask(img, cat='person', instance='#0', show=False)
mask = details.getMask(img, show=False)

mycmap = get_rand_color_map()
mycmap.set_under(alpha=0.0)
nonzero = np.unique(mask[np.nonzero(mask)])
# axarr[0].imshow(mask, cmap=mycmap, vmin=np.min(nonzero), vmax=np.max(nonzero)+1)

# plot point
axarr[0].plot(pt[0], pt[1], 'ro')

plt.show()



# print(type(img))
# print(type(mask))


# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(mask)
# plt.show()


for im, val in data.items():
    fig, ax = plt.subplots(1)
        print(im)
        details.showImg(im, ax=ax, wait=True)
        plot_points(val, ax)