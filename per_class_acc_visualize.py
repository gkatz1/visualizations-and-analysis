import os
import json
import csv
import argparse
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import itertools

CLASS_NAMES = [
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
    'general'
]

CLASSES_WITH_ANNOS = [
      'aeroplane',
      'bicycle',
      'bird',
      'bottle',
      'bus',
      'car',
      'cat',
      'cow',
      'dog',
      'horse',
      'motorbike',
      'person',
      'potted-plant',
      'sheep',
      'train',
      'general'
]



def plot_accuracies(cls, per_class_acc_path, results_summary_path, accuracy_calc_method):
    with open(per_class_acc_path, 'r') as f:
        if accuracy_calc_method == 'mean_per_class':
            # last entry is the mean-per-class
            # for this method we also want to present the per-class accuracy
            data = {int(k):v[-1] for k,v in json.load(f).items()}
        else:
            data = {int(k):v for k,v in json.load(f).items()}

    with open(results_summary_path, 'r') as f:
        total_acc = float(f.read().split('\n')[2].split(' = ')[1])
    if cls:
        print("total accuracy: {} = {}".format(cls, total_acc))
    else:
        print("total accuracy = {}".format(total_acc))

    
    sorted_keys = sorted(data, key=data.get, reverse=True)
    xs = sorted_keys
    print(data)
    print("\n\n\n")
    print(sorted_keys)
    # raise NotImplementedError()

    labels = [CLASS_NAMES[k - 1] for k in sorted_keys]
    pos = np.arange(len(sorted_keys))
    values = [data[k] for k in sorted_keys]


    if cls:
        plt.figure(cls)
    else:
        plt.figure()

    plt.bar(pos, values, align='center', alpha=0.5)
    plt.xticks(pos, labels)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel('accuracy')
    if cls:
        plt.title('{}-model, accuracy breakdown'.format(cls))
    else:
        plt.title('accuracy breakdown')

    
    for i, v in enumerate(values):
        plt.text(i - 0.3, v + 0.01, "{:.3f}".format(v), fontweight='bold')



def plot_bar_graphs(params):
    
    exp_name = params.experiment_name
    class_name = params.class_name
    class_specific_models_mode = params.class_specific_models_mode
    accuracy_calc_method = params.accuracy_calc_method

    if class_specific_models_mode:
        for i, cls in enumerate(CLASS_NAMES):
            class_name = cls
            base_path = '../../../../resources/eval_results/{0}/{0}_{1}/'.format(exp_name, class_name)
            if not os.path.exists(base_path):
                continue
            per_class_acc_path = base_path + 'per_class_acc'
            results_summary_path = base_path + 'summary'
            plot_accuracies(class_name, per_class_acc_path, results_summary_path, accuracy_calc_method)

    else:
        cls = None
        base_path = '../../../../resources/eval_results/{}/'.format(exp_name)
        per_class_acc_path = base_path + 'per_class_acc'
        results_summary_path = base_path + 'summary'
        plot_accuracies(cls, per_class_acc_path, results_summary_path, accuracy_calc_method)


    plt.ion()
    plt.show()
    user_input = raw_input("press any key to quit ")

# TODO: if accuracy_calc_method == 'mean_per_class' also drop file
# for the class-accuracies (object_acc, part_acc)
def make_table(params):
    # work plan
    # for every class - load the accuracies into dictionary
    # dump every dictrionary like this into a different row by some fixed order (e.g. person, animals, transportation, furniture)
    # remember that classes are indexed -1 in here compared to original (no background int the array of CLASS_NAMES)
    
    exp_name = params.experiment_name
    class_name = params.class_name
    class_specific_models_mode = params.class_specific_models_mode
    accuracy_calc_method = params.accuracy_calc_method
    assert accuracy_calc_method in ['mean_per_class', 'object_accuracy',
        'part_accuracy','overall_list_format', 'overall']
    csv_path = '../../docs/{}/{}/class_accuracy_breakdown.csv'.format(exp_name, accuracy_calc_method)
    if not os.path.isdir(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))
    normalize = False
    min_val = 0.15384615384615385
    max_val = 0.967032967032967
    
    header = [''] + CLASSES_WITH_ANNOS
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    accs = list()

    for i, cls in enumerate(CLASSES_WITH_ANNOS):
        class_name = cls
        base_path = '../../../../resources/eval_results/{0}/{0}_{1}/'.format(exp_name, class_name)
        per_class_acc_path = base_path + 'per_class_acc'
        if not os.path.exists(per_class_acc_path):
            continue
        # results_summary_path = base_path + 'summary'  # can add this also somehow (summary column)
    
        with open(per_class_acc_path, 'r') as f:
            if accuracy_calc_method == 'mean_per_class':
            # last entry is the mean-per-class
                data = {int(k):v[-1] for k,v in json.load(f).items()}
            elif accuracy_calc_method == 'object_accuracy':
                data = {int(k):v[0] for k,v in json.load(f).items()}
            elif accuracy_calc_method == 'part_accuracy':
                data = {int(k):v[1] for k,v in json.load(f).items()}
            elif accuracy_calc_method == 'overall_list_format':
                data = {int(k):v[-2] for k,v in json.load(f).items()}
            elif accuracy_calc_method == 'overall':
                data = {int(k):v for k,v in json.load(f).items()}

        if normalize:
            sorted_vals = [((data[k] - min_val) / (max_val - min_val)) * 100 for k in sorted(data)]
        else:
            sorted_vals = [data[k] for k in sorted(data)]
        accs.append(sorted_vals)

        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            to_write = [cls]
            to_write += ['{:.3f}'.format(el) for el in sorted_vals]
            # to_write += sorted_vals
            writer.writerow(to_write)
            
    accs = [l for l in reversed(accs)]
    
    # (x - min) / (max - min)
#    lists_combined = list(itertools.chain.from_iterable(accs))
#    min_val = min(lists_combined)
#    max_val = max(lists_combined)
#    print(min_val, max_val)
#    normalized_accs = list()
    # for l in accs:
        # print(l)
        # print("\n\n\n")
        # HERE - normalize values and dump table

    # plt.imshow(accs, cmap='hot')
    # plt.show()



def main(args):
    exp_name = args.experiment_name
    class_name = args.class_name
    class_specific_models_mode = args.class_specific_models_mode
    action = args.action
    
    if action == 'plot_bar_graphs':
        plot_bar_graphs(args)

    if action == 'make_table':
        make_table(args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--experiment-name', required=True)
    parser.add_argument('-c', '--class-name', default='aeroplane')
    parser.add_argument('-s', '--class-specific-models-mode', action='store_true',
                        help="if experiment directory contains subdirectories, "
                        "where each subdirectory corresponds to results for class"
                        "specfic model evaluation (e.g. sheep-model)")
                        
    parser.add_argument('-am', '--accuracy-calc-method', default='overall')
                        
    parser.add_argument('-a', '--action', default='plot_bar_graphs',
                        help="what function would you like to use")
    
    main(parser.parse_args())











## total accuracies per compress by class
#{'sheep': 0.764184397163, 'horse': 0.752659574468, 'bicycle': 0.600177304965, 'aeroplane': 0.756205673759, 'cow': 0.68085106383, 'sofa': 0.346631205674, 'bus': 0.591312056738, 'dog': 0.725177304965, 'cat': 0.597517730496, 'person': 0.691489361702, 'train': 0.678191489362, 'diningtable': 0.364361702128, 'bottle': 0.416666666667, 'car': 0.587765957447, 'tvmonitor': 0.352836879433, 'chair': 0.348404255319, 'potted-plant': 0.55585106383, 'bird': 0.654255319149, 'boat': 0.369680851064, 'motorbike': 0.635638297872}
