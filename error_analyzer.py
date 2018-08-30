import json
import argparse
import os

# TODO: combine all error types possible (in order to be able to compare them between different models)

# ********* utils *********
def get_objpart_label(x):
    if x <= 0:
        return -1
    if x <= 20:
        return 0
    if x <= 40:
        return 1
    if x <= 60:
        return 2
    return -1

def get_semantic_label(x):
    if x <= 0:
        return -1
    if x <= 20:
        return x
    if x <= 40:
        return x - 20
    if x <= 60:
        return x - 40
    return -1


def print_stat(errors, total_num_errs, property_str):
    num_occ = sum([len(errors[err]) for err in errors if property_str in err])
    print("{} = {:.3f}% ({})\n".format(property_str, float(num_occ) / total_num_errs * 100, num_occ))


def get_error(vals, merge_level):
    
    if merge_level == 'binary':
        gt = vals[0]
        pred = vals[1]
        if gt == pred:
            # return ERROR_TYPE['binary_correct']
            return 'correct' #'binary_correct'
        if gt > pred:
            # gt == 1, pred == 0
            return 'binary, wrong_objpart - object_instead_of_part'
        if pred > gt:
            return 'binary, wrong_objpart - part_instead_of_object'

    if merge_level == '41-way':
        gt = vals[0]
        pred = vals[1]
        gt_objpart_label = get_objpart_label(gt)
        pred_objpart_label = get_objpart_label(pred)
        gt_semantic_label = get_semantic_label(gt)
        pred_semantic_label = get_semantic_label(pred)
        # print("gt: label = {}, objpart = {}, semantic = {}".format(gt, gt_objpart_label, gt_semantic_label))

        if pred == gt:
            return 'correct' #'41-way, correct'

        if gt_semantic_label == pred_semantic_label:
            if gt_objpart_label > pred_objpart_label:
                return '41-way, correct_semantic, wrong_objpart - object_instead_of_part'
            if gt_objpart_label < pred_objpart_label:
                return '41-way, correct_semantic, wrong_objpart - part_instead_of_object'
        else:
            # TODO: the next things is to build a distribution of the
            # semantic classes that were predicted mistakenly for each class
            # e.g. for points with a cat class ground truth: divide the errors by the semantic
            # label of the predicted class (e.g. which points with semantic label of cat
            # were predicted to be dog (and more in depth which were classified with the wrong
            # objpart class & what was the confusion (object instead or part / part instead
            # of object))

            # *********** HERE ***********
            # start with the simple analysis
            # 1. object instead of part
            # 2. part instead of object
            # 3. same objpart cat. but different semantic
            if gt_objpart_label == pred_objpart_label:
                if gt_objpart_label == 0:   # object
                    return '41_way, wrong_semantic, correct_objpart - object'
                if gt_objpart_label == 1:   # part
                    return '41_way, wrong_semantic, correct_objpart - part'
            if gt_objpart_label > pred_objpart_label:
                return '41_way, wrong_semantic, wrong_objpart - object_instead_of_part'
            if gt_objpart_label < pred_objpart_label:
                return '41_way, wrong_semantic, wrong_objpart - part_instead_of_object'


#def semantic_analysis(objpart_pred_file_path, separated_pred_file_path, params):
#    """
#    objpart file - binary / trinary
#    separated file - 41-way / 61-way
#    """
#    # check what is the distribution of points where (binary + semantic == objpart gt)
#    # we have binary file containing both the objpart binary prediction & semantic prediction
#    # combine both and compare to the ground truth label in the 41-way prediction file
#
#    with open(binary_pred_file_path, 'r') as f:
#        objpart_data = json.loads(f.readline().strip())
#
#    with open(separated_pred_file_path, 'r') as f:
#        separated_data = json.loads(f.readline().strip())
#
#    # run through imags, run through each point --> comapre
#    equal_pts = 0  # a points will be called 'equal' iff combination(binary objpart, 21-way semantic) == 41-way objpart label
#    correct_equal_pts = 0  # same as equal but also eqauls to the ground truth 41-way label
#
#    for im, objpart_pts in objpart_data.items():
#        objpart_pts = separated_data[im]
#
#        num_pts += len(objpart_pts)
#        for pt in objpart_pts:
#            objpart_vals = objpart_pts[pt]
#            separated_vals = separated_pts[pt]
#            combined_label = get_combined_label(objpart_vals[0], objpart_vals[3])
#            equal_pts += int(combined_label == separated_vals[1])  # objpart_vals[1] <- [0,1]  objpart_vals[3] <- [0,20]
#            correct_equal_pts += int(combined_label == separated_vals[1] == separated_vals[0])
#
#def get_combined_label(objpart, semantic)
#    """
#    input: 2 different labels
#        objpart - [0,2]
#        semantic - [0,20]
#    output: the combined label of the 2 labels
#    """



def analyze_errors(data, params):
    merge_level = params['merge_level']
    dump_errors_json = params['dump_errors_json']
    json_dump_path = params['json_dump_path']
    
    correct_pts = 0
    num_pts = 0
    errors = dict()
    
    # if merge_level == 'binary':
    for im, pts in data.items():
        num_pts += len(pts)
        print(len(pts), pts)
        for pt, vals in pts.items():
            err = get_error(vals, merge_level)
            if err not in errors:
                errors[err] = {}  # do we also need dictionary for the other direction?
            # errors[err].append({im: pt})
            cur_im_errs = errors[err]
            if im not in cur_im_errs:
                cur_im_errs[im] = []
            cur_im_errs[im].append(pt)
                
    print("\ntotal number of points = {}".format(num_pts))
    total_num_errs = sum([len(errors[err]) for err in errors if err != 'correct'])
    print("total number of errors = {}".format(total_num_errs))
    print("correct predictions = {}\n".format(num_pts - total_num_errs))
    print("accuracy = {:.3f}%\n".format((1 - float(total_num_errs) / num_pts) * 100))

    for err in errors:
        cur_num_errs = len(errors[err])
        if err != 'correct':
            print("{} - {:.3f}% ({})".format(err, float(cur_num_errs) / total_num_errs * 100, cur_num_errs))
    print('')

    # correct_semantic = sum([len(errors[err]) for err in errors if 'correct_semantic' in err])
    # print("correct_semantic = {:.3f}% ({})".format(float(correct_semantic) / total_num_errs * 100 ,correct_semantic))

    # correct_objpart = sum([len(errors[err]) for err in errors if 'correct_objpart' in err])
    # print("correct_objpart = {:.3f}% ({})\n".format(float(correct_objpart) / total_num_errs * 100 ,correct_objpart))

    # part_instead_of_object = sum([len(errors[err]) for err in errors if 'part_instead_of_object' in err])
    # print("part_instead_of_object = {:.3f}% ({})\n".format(float(part_instead_of_object) / total_num_errs * 100 ,part_instead_of_object))
    print("(percentage is computed out of errors only)\n")
    properties_strs = ['correct_semantic',  'wrong_semantic', 'wrong_objpart','correct_objpart', 'part_instead_of_object', 'object_instead_of_part']
    for prop in properties_strs:
        print_stat(errors, total_num_errs, prop)

    if dump_errors_json:
        with open(json_dump_path, 'w') as f:
            json.dump(errors, f)


    # semantic_analysis(data, params)


def main(args):
    # ***** params *****
    exp_name = args.experiment_name
    merge_level = args.merge_level
    assert merge_level in ['binary', 'trinary', '41-way', '61-way']
    dump_errors_json = args.dump_errors_json
    # json_path = '../../json/prediction_jsons/{}_predictions.json'.format(exp_name)
    params['json_path'] = '../../../../resources/eval_results/{}/predictions.json'.format(exp_name)
    base_path = 'errors_dumps/'
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    
    with open(json_path, 'r') as f:
        data = json.loads(f.readline().strip())
    
    params = dict()
    params['merge_level'] = merge_level
    params['dump_errors_json'] = dump_errors_json
    params['json_dump_path'] = os.path.join(base_path, "{}_errors.json".format(exp_name))
    analyze_errors(data, params)

# binary: errors - (1), (2)
# how many of each error
# which images in each error category
#



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameters for controling the pipeline execution")
    parser.add_argument('-n', '--experiment-name', required=True,
                     help="name of experiment to be visualized")
    parser.add_argument('-m', '--merge-level', default='binary',
                        help="type of objpart output (number of objpart classes)")
    parser.add_argument('-d', '--dump-errors-json', action='store_true',
                        help="dump json file of error types")
                        
    args = parser.parse_args()
    main(args)



