# <loader.py>
#
# Functions to load data into memory.
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


def read_ranked_lists_file(file_path, top_k):
    print("\t\t\tReading file", file_path)
    with open(file_path, "r") as f:
        return [[int(y) for y in x.strip().split(" ")][:top_k]
                for x in f.readlines()]


def read_distance_matrix_file(file_path):
    print("\t\t\tReading file", file_path)
    with open(file_path, "r") as f:
        return [[float(y) for y in x.strip().split(" ")]
                for x in f.readlines()]


def read_training_and_test_set_indexes_mnist(file_path):
    with open(file_path, "r") as handle:
        test_indexes = []
        training_indexes = []
        i = 0
        for lb in handle.readlines():
            name = lb.split("_")[0]

            if name == "training":
                training_indexes.append(i)
            elif name == "testing":
                test_indexes.append(i)

            i += 1

        handle.close()

    return training_indexes, test_indexes
