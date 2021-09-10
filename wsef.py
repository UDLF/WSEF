# <wsef.py>
#
# Main implementation file.
#
# @Authors and Contributors:
#       Lucas Pascotti Valem <lucas.valem@unesp.br>
#       João Gabriel Camacho Presotto <joaopresotto@gmail.com>
#       Nikolas Gomes de Sá <NIKOLAS567@hotmail.com>
#       Daniel Carlos Guimarães Pedronette <daniel.pedronette@unesp.br>
#
# ----------------------------------------------------------------------------
#
# This file is part of Weakly Supervised Experiments Framework (WSEF).
# Official Repository: https://github.com/UDLF/WSEF
#
# WSEF is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# WSEF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with WSEF.  If not, see <http://www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------------


import os
import numpy as np
import statistics
import loader
import gc
import utils
import correlation_functions as correlation_func

from pathlib import Path
from pyopf import OPFClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from scipy import sparse
from gcn import GCNClassifier

from config import datasets, dataset_settings


def create_cor_matrix(correlationMeasure, rks, top_k, n, npy_name):
    dir_path = os.path.join("cor_matrices_ws", os.path.join("sparse"))
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        # directory already exists
        pass

    cor_matrix_file = os.path.join(dir_path, npy_name + "_top-k-" + str(top_k))
    cor_matrix_path = Path(cor_matrix_file + ".npz")
    if not cor_matrix_path.is_file():
        # Iterate for each ranked list and compute correlation
        print("\t\t\tComputing correlations...")
        correlationMatrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in rks[i]:
                if correlationMatrix[j][i] != 0:
                    correlationMatrix[i][j] = correlationMatrix[j][i]
                else:
                    correlationMatrix[i][j] = correlationMeasure(rks[i],
                                                                 rks[j],
                                                                 top_k)

        print("\t\t\tExport correlation matrix to numpy file...")
        sparse_cor_matrix = sparse.csr_matrix(correlationMatrix)

        if npy_name != "":
            print("\t\t\tExporting to file", cor_matrix_file)
            sparse.save_npz(cor_matrix_file, sparse_cor_matrix)
            # np.savez_compressed(cor_matrix_file,cor_matrix=correlationMatrix)
    else:
        print("\t\t\tCorrelation Matrix Already Exists Loading...")
        sparse_cor_matrix = sparse.load_npz(cor_matrix_path)

    return sparse_cor_matrix


def run(features,
        labels,
        folds,
        rks,
        classifier="opf"):
    print("\t\t\tRunning Classifier without Expansion ...")

    results = []
    count = 0
    for test_index, train_index in folds:
        print("\t\t\t\tRunning for Fold", count)
        count += 1

        train_features = np.array([features[i] for i in train_index],
                                  dtype=float)
        train_labels = [labels[i] for i in train_index]

        test_features = np.array([features[i] for i in test_index],
                                 dtype=float)
        test_labels = [labels[i] for i in test_index]

        if classifier == "svm":
            clf = svm.SVC(kernel="poly", degree=2, gamma=0.001, C=10)
            # clf = svm.SVC()

            clf.fit(train_features, train_labels)

            pred = clf.predict(test_features)
        elif classifier == "opf":
            # Init OPF
            opf = OPFClassifier()

            train_features = np.float32(train_features)
            test_features = np.float32(test_features)

            # Training
            opf.fit(train_features, train_labels)

            # Predict
            pred = opf.predict(test_features)
        elif classifier == "knn":
            neigh = KNeighborsClassifier(n_neighbors=20)

            neigh.fit(train_features, train_labels)

            pred = neigh.predict(test_features)
        elif classifier == "gcn":
            clf = GCNClassifier(classifier, rks, len(labels),
                                number_neighbors=40)
            clf.fit(test_index, train_index, features, labels)
            pred = clf.predict()
        else:
            print("Classifier not found...")
            exit(1)

        # Append current result
        results.append([pred, test_labels, evaluation(pred, test_labels)])

    return results


def run_ws(features,
           labels,
           cor_matrix,
           thresholds,
           folds,
           rks,
           optimal_threshold_id,
           correlationMeasure,
           results_path,
           reranking=False,
           classifier="opf",
           plot_expansion=False,
           dataset_name="",
           descriptor_name=""):
    print("\t\t\tRunning Classifier with Expansion ...")

    results = []
    train_labels_expansion = []
    train_features_expansion = []
    train_index_expansion = []
    new_elems_class_expansion = []
    # for test_index, train_index in folds:
    for fold_id in range(len(folds)):
        print("\t\t\t\tRunning for Fold", fold_id)
        test_index = folds[fold_id][0]  # 90% of data
        train_index = folds[fold_id][1]  # 10% of data

        train_features = np.array([features[i] for i in train_index],
                                  dtype=float)
        train_labels = [labels[i] for i in train_index]

        test_features = np.array([features[i] for i in test_index],
                                 dtype=float)
        test_labels = [labels[i] for i in test_index]

        # new thrs window=2 test
        th = thresholds
        # **********************
        res_sets = {}
        for i in train_index:
            # Recover all non zero entries in i row
            i_set = [(y, cor_matrix[i][x, y])
                     for x, y in zip(*cor_matrix[i].nonzero())]
            res_sets[i] = set()
            for column, correlation in i_set:
                if correlation >= th:
                    res_sets[i].add(column)

        # Add new elems based on the res_set
        new_elems = []
        new_elems_class = []
        # index of elements indicated by different classes
        indexes_to_remove = []
        for i in train_index:
            for elem in res_sets[i]:
                # do not guess class for already known elements
                if elem in train_index:
                    continue
                if elem not in new_elems:
                    # guess the class label
                    new_elems.append(elem)
                    new_elems_class.append(labels[i])
                else:
                    # remove elements that are in both sets
                    # and do not belong to some guessed class
                    index = new_elems.index(elem)
                    if (new_elems_class[index] != labels[i]) and (
                        index not in indexes_to_remove
                    ):
                        indexes_to_remove.append(index)

        # Remove elements indicated by different classes
        for index in sorted(indexes_to_remove, reverse=True):
            new_elems.pop(index)
            new_elems_class.pop(index)

        # Add new elems to training
        if new_elems != []:
            train_index = np.concatenate([train_index, new_elems])
            train_features = np.concatenate(
                [train_features, [features[i] for i in new_elems]]
            )

            train_labels = np.concatenate([train_labels, new_elems_class])

        if plot_expansion:
            train_labels_expansion.append(train_labels)
            train_features_expansion.append(train_features)
            train_index_expansion.append(train_index)
            new_elems_class_expansion.append(new_elems_class)

        if classifier == "svm":
            clf = svm.SVC(kernel="poly", degree=2, gamma=0.001, C=10)

            clf.fit(train_features, train_labels)
            pred = clf.predict(test_features)
        elif classifier == "opf":
            # Init OPF
            opf = OPFClassifier()

            train_features = np.float32(train_features)
            test_features = np.float32(test_features)

            # Training
            opf.fit(train_features, train_labels)

            # Predict
            pred = opf.predict(test_features)
        elif classifier == "knn":
            neigh = KNeighborsClassifier(n_neighbors=20)

            neigh.fit(train_features, train_labels)

            pred = neigh.predict(test_features)
        elif classifier == "gcn":
            clf = GCNClassifier(classifier, rks, len(labels),
                                number_neighbors=40)
            clf.fit(test_index, train_index, features, labels)
            pred = clf.predict()
        else:
            print("Classifier not found...")
            exit(1)

        # Append current result
        results.append([pred, test_labels, evaluation(pred, test_labels)])

    # **************************************************************
    # Decide the best fold to plot the expansion, plot the before
    # and after the expansion, as well as the original
    # **************************************************************
    if plot_expansion:
        # UMAP Plot
        embedding, path = utils.umap_plot(features, results_path)
        labels = np.array(labels)

        best_fold_id = find_best_fold(results)

        # Original Set
        test_index_original = folds[best_fold_id][0]  # 90% of data
        train_index_original = folds[best_fold_id][1]  # 10% of data
        train_labels_best_original = np.array([labels[i]
                                              for i in train_index_original])

        # Plot Original Set
        # ***
        # Entire dataset
        utils.scatter_full(embedding,
                           labels,
                           dataset_name,
                           descriptor_name,
                           path)
        # 10%/90%
        utils.scatter_fold(
            "initial",
            correlationMeasure,
            classifier,
            best_fold_id,
            embedding,
            train_index_original,
            test_index_original,
            train_labels_best_original,
            labels,
            dataset_name,
            descriptor_name,
            reranking,
            path,
        )

        # Weakly Supervised Set
        train_labels_best_expansion = train_labels_expansion[best_fold_id]
        labeled_indexes = train_index_expansion[best_fold_id]
        new_elems_labels = new_elems_class_expansion[best_fold_id]

        # need to find the unlabeled indexes to pass to plot function
        indexes = np.array(np.array([i for i in range(len(features))]))
        unlabeled_indexes = np.array([i for i in indexes
                                     if i not in labeled_indexes])

        # get real labels and the predicted labels to show accuracy on plot
        real_labels = [labels[i] for i in labeled_indexes
                       if i not in train_index_original]

        # Plot Expanded Set
        utils.scatter_fold(
            "best_fold",
            correlationMeasure,
            classifier,
            best_fold_id,
            embedding,
            labeled_indexes,
            unlabeled_indexes,
            np.array(train_labels_best_expansion),
            labels,
            dataset_name,
            descriptor_name,
            reranking,
            path,
            evaluation(real_labels, new_elems_labels)
        )
    # ******************************************************************************************************************************************************

    return results


def find_best_fold(results):
    acc_values = np.array([acc for _, _, acc in results])
    return np.argmax(acc_values)


def read_groundtruth_file(filepath):
    with open(filepath) as f:
        classes = [int(x.split(":")[-1]) for x in f.readlines()]
    return classes


def convert_labels_to_integers(labels):
    # separate unique classes
    classes_set = set()
    for lb in labels:
        classes_set.add(lb)
    # label classes by integers
    classes_dict = dict()
    for i, c in enumerate(classes_set):
        classes_dict[c] = i
    # rename labels by integers starting from 0
    for i in range(len(labels)):
        labels[i] = classes_dict[labels[i]]
    return labels


def evaluation(pred, labels):
    acc = 0
    n = len(pred)
    for i in range(n):
        if pred[i] == labels[i]:
            acc += 1
    return acc / n


print("* Weakly Supervised Experiments Framework (WSEF) *\n")

# MAIN CODE
# Create Results Folder
results_path = "results"
try:
    os.makedirs(results_path)
except FileExistsError:
    # directory already exists
    pass


# *****
# *****
# Datasets
for dataset in datasets:
    print("Dataset: ", dataset)

    n_folds = dataset_settings[dataset]["n_folds"]
    n_executions = dataset_settings[dataset]["n_executions"]
    classifiers = dataset_settings[dataset]["classifiers"]
    descriptors = dataset_settings[dataset]["descriptors"]
    L = dataset_settings[dataset]["L"]
    top_k = dataset_settings[dataset]["top_k"]
    corMeasures = dataset_settings[dataset]["correlation_measures"]
    dataset_size = dataset_settings[dataset]["dataset_size"]
    # *****
    # Classifiers
    for c in classifiers:
        print("\tClassifier: ", c)

        # Create matrix to store all results
        matrix_length = 1 + (len(corMeasures) * 2)
        results_matrix = [[] for i in range(matrix_length)]

        file = open(
            os.path.join(results_path, str(c) + "_" + str(dataset) + ".txt"),
            "w+",
        )

        print("Classifier:", c, file=file)
        print("Dataset:", dataset, file=file)
        print("Folds:", n_folds, file=file)
        print("n_executions:", n_executions, file=file)
        # *****
        # Descriptors
        for desc in descriptors:
            print("\t\tDescriptor: ", desc)

            # gcns require to load rks for computing the knn graph
            d = "datasets"
            descriptor_path = (
                os.path.join(os.path.join(os.path.join(d, dataset),
                             "rks"), desc) + ".txt"
            )
            # Reading ranked lists
            rks = loader.read_ranked_lists_file(descriptor_path, L)

            # *****
            #   Without WS
            print("\t**********", file=file)
            print("\tDescriptor:", desc, file=file)
            print("\t**********", file=file)

            feature_path = (
                os.path.join(
                    os.path.join(os.path.join("datasets", dataset),
                                 "features"), desc
                )
                + ".npz"
            )
            features = np.load(feature_path)["features"]

            full_gt_path = os.path.join(os.path.join("datasets", dataset),
                                        "groundtruth.txt")
            labels = read_groundtruth_file(full_gt_path)
            labels = convert_labels_to_integers(labels)

            features = np.array(features, dtype=float)

            # Split data in folds
            folds = utils.fold_split(features, labels, n_folds=n_folds)

            acc_without_ws = 0

            print(
                "\tRunning",
                n_executions,
                "times without training set expansion...",
                file=file,
            )

            plot_expansion = True

            acc_list_without_ws = []
            for i in range(n_executions):
                pred = run(
                    features,
                    labels,
                    folds,
                    rks,
                    classifier=c,
                )
                acc_acum = 0
                for p, l, acc in pred:
                    acc_acum += acc

                acc_list_without_ws.append(acc_acum / n_folds * 100)

            acc_without_ws = statistics.mean(acc_list_without_ws)

            print(
                "\t\tMean acc = {}%".format(round(acc_without_ws, 2)),
                file=file,
            )

            results_matrix[0].append(
                round((statistics.mean(acc_list_without_ws)), 2)
            )

            # *****
            # Correlation Measures
            # *****
            result_matrix_index = 1
            for corStr in corMeasures:
                print("\t\t\tCorrelation Measure: ", corStr)

                thresholds = dataset_settings[dataset]["thresholds"][corStr]

                d = "datasets"
                descriptor_path = (
                    os.path.join(os.path.join(os.path.join(d, dataset),
                                 "rks"), desc) + ".txt"
                )
                npy_name = dataset + "_" + desc + "_" + corStr

                npy_name = "sparse_matrix-" + npy_name

                # Get Correlation Measure
                cor_measure = correlation_func.get_correlation_func(corStr)
                # Reading Correlation Matrix or Creating It
                cor_matrix = create_cor_matrix(
                    cor_measure,
                    rks,
                    top_k,
                    dataset_size,
                    npy_name,
                )

                th = thresholds

                # Rounding threshold
                th = round(th, 2)

                acc_list_with_ws = []

                print("\t****", file=file)
                print(
                    "\tCorrelation Measure:",
                    corStr,
                    "/ top_k:",
                    top_k,
                    "/ th:",
                    th,
                    file=file,
                )
                print(
                    "\tRunning",
                    n_executions,
                    "times with training set expansion...",
                    file=file,
                )
                print("\t\tThreshold =", th, file=file)

                for i in range(n_executions):
                    pred = run_ws(
                        features,
                        labels,
                        cor_matrix,
                        th,
                        folds,
                        rks,
                        optimal_threshold_id=str(
                            dataset + "_" + desc + "_" + corStr
                        ),
                        correlationMeasure=corStr,
                        results_path=results_path,
                        classifier=c,
                        plot_expansion=plot_expansion,
                        dataset_name=dataset,
                        descriptor_name=desc,
                    )

                    acc_acum = 0
                    for _, _, acc in pred:
                        acc_acum += acc

                    acc_list_with_ws.append(acc_acum / n_folds * 100)

                acc_with_ws = statistics.mean(acc_list_with_ws)

                # compute gains
                rel_gain = ((acc_with_ws - acc_without_ws)/acc_without_ws)*100
                abs_gain = acc_with_ws - acc_without_ws

                # print info to file
                print(
                    "\t\tMean acc = {}%".format(round(acc_with_ws, 2)),
                    file=file,
                )

                print(
                    "\t\tRelative Gain = {}%".format(round(rel_gain, 2)),
                    file=file,
                )

                print(
                    "\t\tAbsolute Gain = {}%".format(round(abs_gain, 2)),
                    file=file,
                )

                results_matrix[result_matrix_index].append(
                    round((statistics.mean(acc_list_with_ws)), 2)
                )
                results_matrix[result_matrix_index + 1].append(
                    round(acc_with_ws - acc_without_ws, 2)
                )
                result_matrix_index += 2

        print("**********", file=file)
        file.close()
        gc.collect()
