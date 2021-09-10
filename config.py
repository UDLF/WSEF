# <config.py>
#
# Definition of parameters for each dataset.
#
# @Authors and Contributors:
#       Lucas Pascotti Valem <lucas.valem@unesp.br>
#       João Gabriel Camacho Presotto <joaopresotto@gmail.com>
#       Nikolas Gomes de Sá <NIKOLAS567@hotmail.com>
#       Daniel Carlos Guimarães Pedronette <daniel.pedronette@unesp.br>
#
# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------


# list of datasets to run execution
datasets = ["flowers"]

# init dictionary of parameters
dataset_settings = dict()

# flowers dataset parameters
dataset_settings["flowers"] = {"descriptors": ["resnet"],
                               "classifiers": ["opf"],
                               "correlation_measures": ["rbo"],
                               "thresholds": {"intersection": 0.15,
                                              "jaccard": 0.45,
                                              "jaccard_k": 0.30,
                                              "kendalltau": 0.55,
                                              "rbo": 0.20,
                                              "spearman": 0.55},
                               "top_k": 80,
                               "dataset_size": 1360,
                               "L": 400,
                               "n_executions": 1,
                               "n_folds": 10,
}

# corel5k dataset parameters
dataset_settings["corel5k"] = {"descriptors": ["acc", "resnet"],
                               "classifiers": ["opf", "svm", "knn", "gcn"],
                               "correlation_measures": ["intersection", "jaccard", "jaccard_k", "kendalltau", "rbo", "spearman"],
                               "thresholds": {"intersection": 0.45,
                                              "jaccard": 0.40,
                                              "jaccard_k": 0.25,
                                              "kendalltau": 0.50,
                                              "rbo": 0.15,
                                              "spearman": 0.45},
                               "top_k": 100,
                               "dataset_size": 5000,
                               "L": 1000,
                               "n_executions": 1,
                               "n_folds": 10,
}

# cub200 dataset parameters
dataset_settings["cub200"] = {"descriptors": ["resnet152"],
                              "classifiers": ["opf", "svm", "knn", "gcn"],
                              "correlation_measures": ["intersection", "jaccard", "jaccard_k", "kendalltau", "rbo", "spearman"],
                              "thresholds": {"intersection": 0.45,
                                             "jaccard": 0.40,
                                             "jaccard_k": 0.25,
                                             "kendalltau": 0.50,
                                             "rbo": 0.15,
                                             "spearman": 0.45},
                              "top_k": 50,
                              "dataset_size": 11788,
                              "L": 1000,
                              "n_executions": 1,
                              "n_folds": 10,
}
