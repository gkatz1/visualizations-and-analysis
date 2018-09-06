import os
import csv
import argparse
import json
import numpy as np
from collections import Counter

# stats:
# 1) 41-way class distribution
# 2) 21-way class distribution
# 3) trainset class distribution
# 4) valset class distribution
# 5) including / not including ambiguous
# 6) how many of the points with at least 1,2,3,4,5 answers?
# 7) how many with more than 3 answers and consensus? how many do not have consensus
#    meaning they are annotated as ambiguous (in the 'consensus or mask out' mode)
#    7.1) What are the distributions for consensus points ?
# 8) how many are annotated as object, how many as part? (percentage)
# 9) what is the object/part split for each class?
# 10) same as (9) but separately for train/val
# 11) same as (9) for val with/withoud ambiguous points

# for valset - already have the points int the validation set from prediction jsons
# can infer also trainset from this.



# TODO
## 1. points distribution (object/part) for:
#    1. val set
#        1. voting
#        2. consensus_or_ambiguous
#    2. training set (current one)
#        1. need to also dump prediction results for all pts and not just with at least 3 answers
#           becasue current one is for pts with at least 3 answers which means that we will count pts that are
#           in the validation as in the training in the code here
#    3. all data
## 2. point distibution including ambiguous points
#    1. add classes_count_61_way
#    2. return value form get_point_label: for the 61-way return also what class it is

# 3. check if calculation for mask_type of voting ('mode') I do here is the same as in the code of the network
# because in the code (utils/pascal_part.py line 103-110
# & in general - what is being done now for mask_type == 'mode' is it makes sense? Run experiments for some other options
# right now it returns the most_common iff there is only 1 winner in the voting

# Questions
# 1. What do we currently do with points annotated as ambiguous in training/testing? (e.g. answers = (-1, -1, -1))

# Notes:
# 1. because all of these are being calculated - can do some things that saves time manualy, no need to jump over the head


_AMBIGUOUS_VAL = -1

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
    'tvmonitor'
]

CLASS_NAMES_41_WAY = [
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
    'tvmonitor_part'
]



def get_stats_numbers():
    return range(len(_PRINT_FUNCTIONS))


# TODO: logic is not same as in the code for mask_type == 'mode'
# make it same
def get_point_label(answers, labeling_method):
    if len(answers) < 3:
        return -2

    if labeling_method == 'consensus_or_ambiguous':
        answers = np.array(answers)
        if np.all(answers == answers[0]):
            return answers[0]
        else:
            return _AMBIGUOUS_VAL

    elif labeling_method == 'voting':
        ctr = Counter(answers)
        return ctr.most_common()[0][0]

    else:
        raise NotImplementedError("unknown labeling method {}".format(
                                 labeling_method))


def get_semantic_class(label):
    return label if label <= 20 else label - 20


# validation - 2 options: mode, consensus_or_mask_out  --> need another prediciton file for mode evaluation
# training - mode with points with at least 3 answers, mode for all points (and also consensus_or_ambiguous)
# currently models are trained with mode for all points

def print_stats_dataset(data, params):
    labeling_method = params['labeling_method']
    which = params['which']
    results_dir = params['results_dir']
    num_pts = 0
    num_ambiguous_pts = 0
    classes_count = {cls:0 for cls in range(len(CLASS_NAMES))}
    classes_count_41_way = {cls:0 for cls in range(len(CLASS_NAMES_41_WAY))}
    minus_two_ct = 0
    
    print('')
    print("~" * 35)
    
    if which == 'training' or which == 'validation':
        validation_data = params['validation_data']
        validation_pts = dict()     # imid -> [pts]
        for im, pts in validation_data.items():
            validation_pts[im] = []
            for pt_coords, pt_answers in pts.items():
                validation_pts[im].append(pt_coords)

    if which == 'all':
        num_images = len(data)
    elif which == 'validation':
        num_images = len(validation_pts)
    elif which == 'training':
        num_images = len(data) - len(validation_pts)

    print("1. number of images annotated: {}\n".format(num_images))

    num_pts_validation = 0
    for im, pts in data.items():
        for pt_coords, pt_answers in pts.items():
            if which == 'validation':
                if im in validation_pts:
                    if pt_coords in validation_pts[im]:
                        label = get_point_label(pt_answers, labeling_method)
                    else:
                        continue
                else:
                    continue
                # label = pt_answers[0]
                # print(pt_coords, label, pt_answers)
            elif which == 'all':
                label = get_point_label(pt_answers, labeling_method)
                cur_num_pts = len(pts.keys())
            elif which == 'training':
                # can first make dictionary of all points in the val set
                # than go through all points and if the points is in the valset don't count it
                # not exact - because some of the points which are in the original validation set
                # are not being used becasue of the use of unambiguous points only
                # so use a file of experiment evaluated with ambiguous points also
                if im in validation_pts:
                    if pt_coords in validation_pts[im]:
                        num_pts_validation += 1
                        continue
                    continue
                label = get_point_label(pt_answers, labeling_method)
            else:
                raise NotImplementedError("unrecognized 'which' dataset option {}".format(
                                          which))
            num_pts += 1
            if label == _AMBIGUOUS_VAL:
                num_ambiguous_pts += 1
            elif label == -2:
                minus_two_ct += 1
            else:
                cls = get_semantic_class(label)
                classes_count[cls] += 1
                classes_count_41_way[label] += 1

    num_pts = num_pts - minus_two_ct
    if which == 'training':
        num_pts -= num_pts_validation
    num_unambiguous_pts = num_pts - num_ambiguous_pts


    print("2. number of points annotated: {}\n".format(num_pts))
    
    print("3. average number of points per image: {:.3f}\n".format(
        float(num_pts) / num_images))

    print("4. number of ambiguous points = {} ({:.3f}%)\n".format(
        num_ambiguous_pts, (float(num_ambiguous_pts) / num_pts) * 100))
        
    print("5. number of unambiguous points = {} ({:.3f}%)\n".format(
        num_unambiguous_pts, (float(num_unambiguous_pts) / num_pts) * 100))

    print("6. distribution of points by classes (%):\n")
    with open(os.path.join(results_dir, 'class_distribution.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'percentage'])

    with open(os.path.join(results_dir, 'class_counts.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'count'])
    
    total_fraction = 0
    for cls, count in classes_count.items():
        print("{} = {}".format(CLASS_NAMES[cls], count))
        with open(os.path.join(results_dir, 'class_counts.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([CLASS_NAMES[cls], '{}'.format(count)])

        fraction = float(count) / num_pts
        total_fraction += fraction
        print("    {} = {:.3f}".format(CLASS_NAMES[cls], fraction * 100))
        with open(os.path.join(results_dir, 'class_distribution.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([CLASS_NAMES[cls], '{:.3f}'.format(fraction * 100)])

    print('')
    # print('{}'.format(total_fraction))



    # print(total_fraction)

    print("7. distribution of points by classes, out of unambiguous points (%):\n")
                            
    total_fraction = 0
    for cls, count in classes_count.items():
        fraction = float(count) / num_unambiguous_pts
        total_fraction += fraction
        print("    {} = {:.3f}".format(CLASS_NAMES[cls], fraction * 100))
    print('')


    print("8. distribution of points by 41 classes (%):\n")
    total_fraction = 0
    for cls, count in classes_count_41_way.items():
        fraction = float(count) / num_pts
        total_fraction += fraction
        print("    {} = {:.3f}".format(CLASS_NAMES_41_WAY[cls], fraction * 100))
    print('')

    print("9. object vs part (%):\n")
    total_fraction = 0
    num_pts_as_objects = sum([classes_count_41_way[cls] for cls in
                              classes_count_41_way if cls <= 20])
    num_pts_as_parts = sum([classes_count_41_way[cls] for cls
                              in classes_count_41_way if cls > 20])

    print("    object = {}".format(float(num_pts_as_objects) / num_pts * 100))
    print("    part = {}".format(float(num_pts_as_parts) / num_pts * 100))


    print("\n")


    with open(os.path.join(results_dir, 'class_counts_breakdown.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'object', 'part'])
    
    # TODO: wrong calculation for the which == validation
    # 10. for each class -> what is the distribution of object/part?
    print("10. per class object/part distribution (%)")
    with open(os.path.join(results_dir, 'class_counts_breakdown.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'object', 'part'])
        # add also ambiguous for the consensus_or_ambiguous mode
        
    for cls_idx, cls in enumerate(CLASS_NAMES):
        if classes_count[cls_idx] == 0:
            continue
        with open(os.path.join(results_dir, 'class_counts_2.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([CLASS_NAMES[cls_idx], classes_count_41_way[cls_idx], classes_count_41_way[cls_idx + 20]])

        object_fraction = float(classes_count_41_way[cls_idx]) / classes_count[cls_idx] * 100
        part_fraction = float(classes_count_41_way[cls_idx + 20]) / classes_count[cls_idx] * 100
        print("{}:".format(cls))
        print("\tobject = {:.2f}".format(object_fraction))
        print("\tpart = {:.2f}".format(part_fraction))
        print('')
        with open(os.path.join(results_dir, 'object_part_distribution.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([cls, '{:.3f}'.format(object_fraction), '{:.3f}'.format(part_fraction)])


_PRINT_FUNCTIONS = [
    print_stats_dataset
]



def main(args):
    stats_numbers = get_stats_numbers()
    
    params = dict()
    params['labeling_method'] = args.labeling_method
    mask_type = args.mask_type_for_validation
    params['which'] = args.which_data
    assert params['which'] in ['validation', 'training', 'all']
    params['results_dir'] = '../../docs/data_stats/{}'.format(params['which'])
    if not os.path.isdir(params['results_dir']):
        os.makedirs(params['results_dir'])
    
    data_json_path = '../../json/pascal_gt.json'

    if params['which'] == 'validation':
        params['labeling_method'] = mask_type
        assert mask_type in ['consensus_or_ambiguous', 'voting']
        if mask_type == 'consensus_or_ambiguous':
            print("consensus_or_ambiguous")
            val_data_json_path = '../../json/prediction_jsons/exp_37_predictions.json'
        else:
            print("voting")
            # voting ('mask_type == mode' in the code), using only pts with at least 3 answers
            val_data_json_path = '../../json/prediction_jsons/exp_98_voting_at_least_3_answers_predictions.json'
            # data_json_path = '../../json/prediction_jsons/exp_37_predictions.json'

        with open(val_data_json_path, 'r') as f:
            val_data = json.loads(f.readline().strip())
            params['validation_data'] = val_data

    elif params['which'] == 'training':
        data_json_path = '../../json/pascal_gt.json'
        # val_data_json_path = '../../json/prediction_jsons/exp_37_predictions.json'
        val_data_json_path = '../../json/prediction_jsons/exp_98_voting_at_least_3_answers_predictions.json'
        with open(val_data_json_path, 'r') as f:
            val_data = json.loads(f.readline().strip())
        params['validation_data'] = val_data

    
    with open(data_json_path, 'r') as f:
        data = json.loads(f.readline().strip())

    for i in stats_numbers:
        _PRINT_FUNCTIONS[i](data, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--labeling-method', default='consensus_or_ambiguous',
                        help="method used for coverting annotation answers into a label")
    parser.add_argument('-w', '--which-data', default='all',
                        help="which dataset would you like to analyze")
    parser.add_argument('-m', '--mask-type-for-validation', default='consensus_or_ambiguous')
    args = parser.parse_args()
    main(args)






