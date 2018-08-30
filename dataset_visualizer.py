import json
from detail import Detail
from matplotlib import pyplot as plt
import numpy as np
from cycler import cycler
from collections import Counter
import os
import csv
from math import sqrt


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


def get_class(idx):
    return CLASS_NAMES[idx]


# image -> point1 -> [label1, label2, ...]
#       -> point2 -> [label1, label2, ...]
#       .
#       .
#       .
#       -> pointK -> [label1, label2, ...]


# Convert x_y --> point(x, y)
class Point(object):
    # def __init__(self, x_y = None, x = 0, y = 0): # problematic constructore for cases like Point(1, 2) (takes 1 as x_y)
    def __init__(self, *args):
        if len(args) is 1:
            self.x, self.y = self.parse_x_y_str(args[0])
        
        if len(args) is 2:
            self.x = args[0]
            self.y = args[1]
    
    def parse_x_y_str(self, x_y, delim = '_'):
        """asserting x_y is of the form e.g. '129_493'
        """
        x_y = x_y.split(delim)
        return int(x_y[0]), int(x_y[1])

    def to_tuple(self):
        return (self.x, self.y)


# ***************** params *****************
ann_file = '../../json/trainval_merged.json'
im_dir = '../../VOCdevkit/VOC2010/JPEGImages'
pahse = 'trainval'
json_path = '../../json/pascal_gt.json'
log_table_base_path = '../../docs/'
log_table_name= 'log_table.csv'

QUIT = 'q'

line = None                 # TBC
labels = None               # TBC
annot = None                # TBC
ax = None                   # TBC
fig = None                  # TBC
txts = dict()               # TBC
points_clicked = list()     # TBC
cur_points = list()     # TBC


# ***************** hover functionality *****************
def update_annot(ind):
    x, y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = txts['{}_{}'.format(x[ind["ind"][0]], y[ind["ind"][0]])]
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = line.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


# ***************** clicking points *****************
def euclidean_dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def nearest_point(pt):
    """ return cloest point to pt from cur_points
    """
    nearest_point = None
    min_dist = float("inf")
    for p in cur_points:
        dist = euclidean_dist(pt, p.to_tuple())
        if dist < min_dist:
            min_dist, nearest_point = dist, p

    return nearest_point.to_tuple()


def onclick(event):
    global points_clicked
    if event.inaxes == ax:
        cont, _ = line.contains(event)
        if cont:
            ix, iy = event.xdata, event.ydata
            ix, iy = nearest_point((ix, iy))    # get ix, iy of nearest closest point
            if (ix, iy) not in points_clicked:
                points_clicked.append((ix, iy))
                print("- point ({}, {}), # {}".format(ix, iy, len(points_clicked)))

def get_coords_to_write(fig):
    """ get the coords of points which the user would want to save to the log
    """
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    raw_input("Start clicking, press any key when finished: \n")
    fig.canvas.mpl_disconnect(cid)


def log_writer(fig, im):
    yn = raw_input("Would you like to write a note? (y/n): ")
    if yn == 'y':
        global points_clicked
        
        get_coords_to_write(fig)
        write_to_log_table_wrapper(im, points_clicked)      # points are of format [(x1, y1), (x2, y2), ...]
        points_clicked = []


# ***************** manage log table *****************
def write_to_log_table_wrapper(image_name, points_coords):
    notes = raw_input("Enter your note: ")
    write_to_log_table(image_name, notes, points_coords)


def write_to_log_table(image_name, notes, points_coords):
    """ log table: image_name, notes, points coords (coordinations of related points)
    """
    if not os.path.exists(log_table_base_path):
        os.makedirs(log_table_base_path)
    
    full_path = log_table_base_path + log_table_name

    if not os.path.exists(full_path):
        headline = ['image name', 'notes', 'points coords']
        with open(full_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headline)
    
    # add numering (in the title) for each point?
    with open(full_path, 'a') as f:
        writer = csv.writer(f)
        row = [image_name, notes] + ['({}, {})'.format(point_coord[0], point_coord[1]) for point_coord in points_coords]
        writer.writerow(row)


# ***************** plotting *****************
def plot_points(pts, _fig=None, _ax=None):
    global line    # TBC
    global labels  # TBC
    global annot   # TBC
    global ax      # TBC
    global fig     # TBC
    global cur_points
    ax = _ax
    fig = _fig
    cur_points = []
    
    xs = list()
    ys = list()
    
    print('****************\n')
    for pt_coords in pts.keys():
    # for pt_coords in zip(sorted(pts.keys())) # not exactly like this but we want to have 3 dictionaries and add all 3 to
        print('~~~~~~~~~~~~~~~')
        cur_point = Point(pt_coords)

        cur_points.append(cur_point)       # for the use of onclick (note logging)
        
        # xs.append(cur_point.x)
        # ys.append(cur_point.y)
        plot_point(cur_point, pts[pt_coords], ax, xs, ys)

    line, = plt.plot(xs, ys, marker="o", color='r', alpha=0.7, linewidth=0.0001)
    annot = ax.annotate("", xy=(0,0), xytext=(-20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"), arrowprops = dict(arrowstyle="->"))
    annot.set_visible(False)
                        
                        
def plot_point(point, labels, ax, xs, ys):
    """Input: point - Point object
              labels - labels for the point
              ax - plot axes
    """
    global txts
    
    print('(x, y) = ({}, {})'.format(point.x, point.y))
    # ax.plot(point.x, point.y, 'ro')
    xs.append(point.x)
    ys.append(point.y)
    counter = Counter(labels)

    txt = ''
    for key, count in counter.items():
        print ('{} ==> {}').format(key, count)
        txt = txt + '(' + get_class(key) + ', ' + str(count) + '), '
    txt = txt[:-2]
    print('~~~~~~~~~~~~~~~\n\n')
    # ax.annotate(txt, (point.x,point.y))
    txts['{}_{}'.format(point.x, point.y)] = txt


# ***************** view options *****************
def show_1_image(im):
    with open(json_path, 'r') as f:
        data = json.loads(f.readline().strip())

    details = Detail(ann_file, im_dir, pahse)

    while True:
        try:
            val = data[im]
            
            fig, ax = plt.subplots()
            plt.ion()
            plt.show()
            im = str(im)
            details.showImg(im, ax=ax, wait=True)
            ax.set_title(im)
            plot_points(val, fig, ax)
            fig.canvas.mpl_connect("motion_notify_event", hover)
            plt.draw()
            plt.pause(0.001)

            log_writer(fig, im)

        except KeyError:
            print("[-] Error: Invalid image name")

        while True:
            user_input = raw_input("n (another image), c (close all figures), q (quit): ")
            if user_input == 'q':
                return
            if user_input == 'n':
                im = raw_input("Enter image name: ")
                break
            if user_input == 'c':
                plt.close('all')


def show_from_point_on(start_image, index_mode=False):
    """index_mode == True --> start_image is the index of the starting point by index
    """
    
    with open(json_path, 'r') as f:
        data = json.loads(f.readline().strip())

    details = Detail(ann_file, im_dir, pahse)

    found_starting_point = False

    for idx, (im, val) in enumerate(data.items()):
        if not found_starting_point:
            if index_mode:
                if idx >= start_image:
                    found_starting_point = True
                else:
                    continue
            else:
                if im == start_image:
                    found_starting_point = True
                else:
                    continue

        while True:
            input = raw_input('\nn (next image), c (close all figures), q (quit): ')
            if input == 'c':
                plt.close('all')
            if input == 'n':
                break
            if input == 'q':
                return

        fig, ax = plt.subplots()
        plt.ion()
        plt.show()
        im = str(im)
        details.showImg(im, ax=ax, wait=True)
        ax.set_title(im)
        plot_points(val, fig, ax)
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.draw()
        plt.pause(0.001)

        log_writer(fig, im)


def show_all_images():
    with open(json_path, 'r') as f:
        data = json.loads(f.readline().strip())

    details = Detail(ann_file, im_dir, pahse)

    for im, val in data.items():
#        input = raw_input('\nnext image? (y/n): ')
#        if input in ['n', 'no']:
#            while True:
#                user_in = raw_input("Enter 'q' to quit: ")
#                if user_in in ['q', 'quit']:
#                    return
        while True:
            input = raw_input('\nn (next image), c (close all figures), q (quit): ')
            if input == 'c':
                plt.close('all')
            if input == 'n':
                break
            if input == 'q':
                return

        fig, ax = plt.subplots()
        plt.ion()
        plt.show()
        im = str(im)
        details.showImg(im, ax=ax, wait=True)
        ax.set_title(im)
        plot_points(val, fig, ax)
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.draw()
        plt.pause(0.001)

        log_writer(fig, im)

# add a button for writing note
# when clicked -> would give you plave to write the note
# Will tell you to click points
# Would change it's title(button's title) to "click me when finished"
# When finished -> will save the annotation to log file



# ***************** User Input *****************
_SHOW_1_IMAGE = 3
_SHOW_FROM_POINT = 2
_SHOW_ALL_IMAGES = 1

def handle_input(user_in):
    if user_in[0] == _SHOW_1_IMAGE:
        show_1_image(user_in[1])
    
    if user_in[0] == _SHOW_FROM_POINT:
        show_from_point_on(user_in[2], user_in[1])

    if user_in[0] == _SHOW_ALL_IMAGES:
        show_all_images()


def get_user_input():
    while True:
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Welcome to Pascal Points Dataset")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("1. Show all images\n2. Show all images starting from specific location\n"
              "3. Show 1 image only\n")
        while True:
            ans = raw_input("Your chioce: ")
            try:
                ans = int(ans)
                if ans in [1,2,3]:
                    break
            except:
                continue

        if ans in [1, 2, 3]:
            ret = [ans]
            if ans in [2,3]:
                if ans == 2:
                    ch = raw_input("1 (image by index) 2 (Image by name): ")
                    ch = int(ch)
                    index_mode = True if ch == 1 else False
                    ret.append(index_mode)      # index_mode
                    if index_mode:
                        ans = int(raw_input("Enter image index: "))      # by yhe order of the loop of all_images option
                    ret.append(ans)
                    return ret
                # Assuming valid input
                ans = raw_input("Enter image name: ")
                if '.jpg' in ans:
                    ans = ans[:-4]
                if '.jpeg' in ans:
                    ans = ans[:-5]
                ret.append(ans)
            return ret


# ***************** main *****************
def main():
    user_in = get_user_input()
    handle_input(user_in)
    # write_to_log_table('123', 'blah', [(12, 13), (13, 15)])


if __name__ == '__main__':
    main()
















# ***********************************
# Archive

# for key, val in data.items():
#     print 'key = {}'.format(key)
#         print 'val = {}'.format(val)
#         for k in val.keys():
#             pass
#     # print k, '-->', val[k]
#     print('******')
# print("###")


# *********
#def get_points_and_labels(pts_and_vals):
#    """input: pts_and_vals -> dictionary with keys 'x_y' and vals which indicates the values annotated
#        """
#    res = {}
#    for point_coords in pts_and_vals.keys():
#        for label in pts_and_vals[point_coords]:
#            res[Point(point_coords)] = label
#            print(res[Point(point_coords)])
#        print(res[Point(point_coords)])


# *********
#
#with open(json_path, 'r') as f:
#    data = json.loads(f.readline().strip())
#    
#    details = Detail(ann_file, im_dir, pahse)
#    
#    for im, val in data.items():
#        input = raw_input('\nnext image? (y/n): ')
#        if input in ['n', 'no']
#            break
#        
#        fig, ax = plt.subplots()
#        plt.ion()
#        plt.show()
#        im = str(im)
#        details.showImg(im, ax=ax, wait=True)
#        plot_points(val, fig, ax)
#        fig.canvas.mpl_connect("motion_notify_event", hover)
#        plt.draw()
#        plt.pause(0.001)
