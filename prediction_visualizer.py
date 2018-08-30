import json
import argparse
import os
from detail import Detail
from matplotlib import pyplot as plt
import numpy as np
from cycler import cycler
from collections import Counter
import os
import csv
from math import sqrt


CLASSES = [
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
    'tvmonitor'
]

CLASS_NAMES_61 = ['background',
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

TRINARY_CLASS_NAMES = ['object',
                       'part',
                       'ambiguous']


def get_class(idx, merge_level):
    if merge_level == 'binary' or merge_level == 'trinary':
        return TRINARY_CLASS_NAMES[idx]
    else:
        return CLASS_NAMES_61[idx]


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


QUIT = 'q'

lines = None                 # TBC
labels = None               # TBC
annot = None                # TBC
ax = None                   # TBC
fig = None                  # TBC
txts = dict()               # TBC
points_clicked = list()     # TBC
cur_points = list()     # TBC
correctness = dict()


# ***************** hover functionality *****************
def update_annot(ind, correct_pt):
    if correct_pt:
        x, y = lines[0].get_data()
    else:
        x, y = lines[1].get_data()

    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = txts['{}_{}'.format(x[ind["ind"][0]], y[ind["ind"][0]])]
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    
    vis = annot.get_visible()
    if event.inaxes == ax:
        coorect_cont, correct_ind = lines[0].contains(event)
        wrong_cont, wrong_ind = lines[1].contains(event)
        if coorect_cont or wrong_cont:
            if coorect_cont:
                update_annot(correct_ind, True)
            else:
                update_annot(wrong_ind, False)
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
        cont_correct, _ = lines[0].contains(event)
        cont_wrong, _ = lines[1].contains(event)

        if cont_correct or cont_wrong:
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


def log_writer(fig, im, log_table_path):
    yn = raw_input("Would you like to write a note? (y/n): ")
    if yn == 'y':
        global points_clicked
        
        get_coords_to_write(fig)
        # points are of format [(x1, y1), (x2, y2), ...]
        write_to_log_table_wrapper(im, points_clicked, log_table_path)
        points_clicked = []


# ***************** manage log table *****************
def write_to_log_table_wrapper(image_name, points_coords, log_table_path):
    notes = raw_input("Enter your note: ")
    write_to_log_table(image_name, notes, points_coords, log_table_path)


def write_to_log_table(image_name, notes, points_coords, log_table_path):
    """ log table: image_name, notes, points coords (coordinations of related points)
    """
    global correctness
    
    if not os.path.exists(os.path.dirname(log_table_path)):
        os.makedirs(os.parh.dirname(log_table_path))
    
    # full_path = log_table_base_path + log_table_name

    if not os.path.exists(log_table_path):
        headline = ['image name', 'notes', 'points coords']
        with open(log_table_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headline)

    # add numering (in the title) for each point?
    with open(log_table_path, 'a') as f:
        writer = csv.writer(f)
        row = [image_name, notes] + ['({}, {}) - {}\n{}'.format(x, y, "correct" if correctness[(x,y)] else "wrong",
                                                                txts["{}_{}".format(x, y)]) for x, y in points_coords]
        for p in cur_points:
            print("p.x, p.y = {}, {}".format(p.x, p.y))
        writer.writerow(row)
        correctness = {}


# ***************** plotting *****************
def plot_points(pts, _fig=None, _ax=None, merge_level='41-way'):
    global lines    # TBC
    global labels  # TBC
    global annot   # TBC
    global ax      # TBC
    global fig     # TBC
    global cur_points
    
    ax = _ax
    fig = _fig
    cur_points = []
    
    xs_correct = list()
    ys_correct = list()
    xs_wrong = list()
    ys_wrong = list()
    xs = [xs_correct, xs_wrong]
    ys = [ys_correct, ys_wrong]

    
    print('****************\n')
    for pt_coords in pts.keys():
    # for pt_coords in zip(sorted(pts.keys())) # not exactly like this but we want to have 3 dictionaries and add all 3 to
        print('~~~~~~~~~~~~~~~')
        cur_point = Point(pt_coords)

        cur_points.append(cur_point)       # for the use of onclick (note logging)
        
        # xs.append(cur_point.x)
        # ys.append(cur_point.y)
        plot_point(cur_point, pts[pt_coords], ax, xs, ys, merge_level)

    line_correct, = plt.plot(xs[0], ys[0], marker="o", color='g', alpha=0.7, linewidth=0.0001)
    line_wrong, = plt.plot(xs[1], ys[1], marker="o", color='r', alpha=0.7, linewidth=0.0001)
    lines = [line_correct, line_wrong]

    annot = ax.annotate("", xy=(0,0), xytext=(-20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"), arrowprops = dict(arrowstyle="->"))
                        
    annot.set_visible(False)


# add also gt
def plot_point(point, labels, ax, xs, ys, merge_level):
    """Input: point - Point object
              labels - labels for the point
              ax - plot axes
    """
    global txts
    global correctness
    
    print('(x, y) = ({}, {})'.format(point.x, point.y))
    # ax.plot(point.x, point.y, 'ro')
    if labels[0] == labels[1]:
        correctness[(point.x, point.y)] = True
        xs[0].append(point.x)
        ys[0].append(point.y)
    else:
        correctness[(point.x, point.y)] = False
        xs[1].append(point.x)
        ys[1].append(point.y)

    txt = 'gt = {}\npred = {}'.format(get_class(labels[0], merge_level), get_class(labels[1], merge_level))

    print('~~~~~~~~~~~~~~~\n\n')
    # ax.annotate(txt, (point.x,point.y))
    txts['{}_{}'.format(point.x, point.y)] = txt


def get_semantic_class(objpart_class):
    """"
    returns the semantic label of the objparp
    e.g. get_semantic_class(21) --> 1
         get_semantic_class(1) --> 1
    """
    return objpart_class if objpart_class <= 20 else objpart_class - 20



def contains_error(pts, cls):
    for pt, vals in pts.items():
        print(cls, vals[0], vals[1])
        if get_semantic_class(vals[0]) == cls and vals[1] != vals[0]:
            print("contains error! class = {}, gt = {}, pred = {}".format(cls, vals[1], vals[0]))
            return True

def contains_class(pts, cls):
    for pt, vals in pts.items():
        if get_semantic_class(vals[0]) == cls:
            return True

# ***************** view options *****************
def show_1_image(im, params):
    json_path = params['json_path']
    ann_file = params['ann_file']
    im_dir = params['im_dir']
    phase = params['phase']
    log_table_path = params['log_table_path']
    merge_level = params['merge_level']
    
    with open(json_path, 'r') as f:
        data = json.loads(f.readline().strip())

    details = Detail(ann_file, im_dir, phase)

    while True:
        try:
            val = data[im]
            
            fig, ax = plt.subplots()
            plt.ion()
            plt.show()
            im = str(im)
            details.showImg(im, ax=ax, wait=True)
            ax.set_title(im)
            plot_points(val, fig, ax, merge_level)
            fig.canvas.mpl_connect("motion_notify_event", hover)
            plt.draw()
            plt.pause(0.001)

            log_writer(fig, im, log_table_path)

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


def show_from_point_on(start_image, index_mode=False, params={}):
    """index_mode == True --> start_image is the index of the starting point by index
    """
    json_path = params['json_path']
    ann_file = params['ann_file']
    im_dir = params['im_dir']
    phase = params['phase']
    log_table_path = params['log_table_path']
    merge_level = params['merge_level']

    with open(json_path, 'r') as f:
        data = json.loads(f.readline().strip())

    details = Detail(ann_file, im_dir, phase)

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
        plot_points(val, fig, ax, merge_level)
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.draw()
        plt.pause(0.001)

        log_writer(fig, im, log_table_path)


def show_all_images(params):
    json_path = params['json_path']
    ann_file = params['ann_file']
    im_dir = params['im_dir']
    phase = params['phase']
    log_table_path = params['log_table_path']
    merge_level = params['merge_level']

    
    with open(json_path, 'r') as f:
        data = json.loads(f.readline().strip())

    details = Detail(ann_file, im_dir, phase)

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
        plot_points(val, fig, ax, merge_level)
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.draw()
        plt.pause(0.001)

        log_writer(fig, im, log_table_path)


def show_all_errors_for_class(cls, params):
    json_path = params['json_path']
    ann_file = params['ann_file']
    im_dir = params['im_dir']
    phase = params['phase']
    log_table_path = params['log_table_path']
    merge_level = params['merge_level']
    
    print("showing errors for class {}".format(cls))
    
    with open(json_path, 'r') as f:
        data = json.loads(f.readline().strip())

    details = Detail(ann_file, im_dir, phase)

    for im, val in data.items():
    
        # TODO: implement contains_error for json with just 0,1 predictions and annotations
        # do it by using the pascal_gt.json for getting the ground truth class
        # do same in visualize errors by class
        
        if not contains_error(val, cls):
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
        plot_points(val, fig, ax, merge_level)
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.draw()
        plt.pause(0.001)

        log_writer(fig, im, log_table_path)



def show_all_images_containing_given_class(cls, params):
    json_path = params['json_path']
    ann_file = params['ann_file']
    im_dir = params['im_dir']
    phase = params['phase']
    log_table_path = params['log_table_path']
    merge_level = params['merge_level']
    
    print("showing images for class {}".format(cls))
    
    with open(json_path, 'r') as f:
        data = json.loads(f.readline().strip())

    details = Detail(ann_file, im_dir, phase)

    for im, val in data.items():
    
        if not contains_class(val, cls):
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
        plot_points(val, fig, ax, merge_level)
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.draw()
        plt.pause(0.001)
        
        log_writer(fig, im, log_table_path)


# add a button for writing note
# when clicked -> would give you plave to write the note
# Will tell you to click points
# Would change it's title(button's title) to "click me when finished"
# When finished -> will save the annotation to log file



# ***************** User Input *****************
_SHOW_ALL_IMAGES = 1
_SHOW_FROM_POINT = 2
_SHOW_1_IMAGE = 3
_SHOW_ALL_ERRORS_FOR_CLASS = 4
_SHOW_ALL_IMGAES_OF_CLASS = 5

def handle_input(user_in, params):
    if user_in[0] == _SHOW_1_IMAGE:
        show_1_image(user_in[1], params)
    
    if user_in[0] == _SHOW_FROM_POINT:
        show_from_point_on(user_in[2], user_in[1], params)

    if user_in[0] == _SHOW_ALL_IMAGES:
        show_all_images(params)

    if user_in[0] == _SHOW_ALL_ERRORS_FOR_CLASS:
        show_all_errors_for_class(user_in[1], params)

    if user_in[0] == _SHOW_ALL_IMGAES_OF_CLASS:
        show_all_images_containing_given_class(user_in[1], params)


def get_user_input():
    while True:
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Welcome to Pascal Points Dataset")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("1. Show all images\n2. Show all images starting from specific location\n"
              "3. Show 1 image only\n4. Show errors for a given class\n"
              "5. Show all images containing a given class")
        while True:
            ans = raw_input("Your chioce: ")
            try:
                ans = int(ans)
                if ans in [1,2,3,4,5]:
                    break
            except:
                continue

        if ans in [1, 2, 3, 4, 5]:
            ret = [ans]
            if ans in [2,3]:
                if ans == 2:
                    while True:
                        ch = raw_input("1 (image by index) 2 (Image by name): ")
                        if ch in ['1', '2']:
                            ch = int(ch)
                            index_mode = True if ch == 1 else False
                            break
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
            if ans in [4,5]:
                while True:
                    cls = raw_input("enter class name: ")
                    if cls in CLASSES:
                        break
                ret.append(CLASSES.index(cls))

            return ret


# ***************** main *****************
def main(args):
    # ******* params *******
    exp_name = args.experiment_name
    merge_level = args.merge_level
    assert merge_level in ['binary', 'trinary', '41-way', '61-way']
    
    params = dict()
    params['ann_file'] = '../../json/trainval_merged.json'
    params['im_dir'] = '../../VOCdevkit/VOC2010/JPEGImages'
    params['phase'] = 'trainval'
    params['json_path'] = '../../json/prediction_jsons/{}_predictions.json'.format(exp_name)
    # params['json_path'] = '../../../../resources/eval_results/{}/predictions.json'.format(exp_name)
    log_table_base_path = '../../docs/'
    log_table_name = '{}_log_table.csv'.format(exp_name)
    params['log_table_path'] = os.path.join(log_table_base_path, log_table_name)
    params['merge_level'] = merge_level
    
    user_in = get_user_input()
    handle_input(user_in, params)

    # write_to_log_table('123', 'blah', [(12, 13), (13, 15)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameters for controling the pipeline execution")
    parser.add_argument('-n', '--experiment-name', required=True,
                        help="name of experiment to be visualized")
    parser.add_argument('-m', '--merge-level', default='41-way',
                        help="type of objpart output (number of objpart classes)")
                        
    args = parser.parse_args()
    main(args)
















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
#    details = Detail(ann_file, im_dir, phase)
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
