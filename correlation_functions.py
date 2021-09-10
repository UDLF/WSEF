# <correlation_functions.py>
#
# Implementation of correlation functions.
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


import math
import numpy as np


def get_correlation_func(correlation_measure):
    if correlation_measure == "jaccard":
        return compute_jaccard

    if correlation_measure == "jaccard_k":
        return compute_jaccard_k

    if correlation_measure == "rbo":
        return compute_rbo

    if correlation_measure == "kendalltau":
        return compute_kendalltau

    if correlation_measure == "kendallw":
        return compute_kendallw

    if correlation_measure == "spearman":
        return compute_spearman

    if correlation_measure == "intersection":
        return compute_intersection

    print("\n ERROR: Unknown correlation measure:", correlation_measure)
    exit(1)


def compute_jaccard(x, y, top_k):
    return len(set(x[:top_k])
               & set(y[:top_k])) / len(set(x[:top_k]) | set(y[:top_k]))


def get_index(i, x):
    """
    Returns the position of the element 'i' in the ranked list 'x'
    """
    if i in x:
        return x.index(i)
    else:
        return len(x)


def check_sizes(x, y):
    """
    Verifies if the ranked lists 'x' and 'y' have the same size
    """
    if len(x) != len(y):
        return False
    return True


def compute_kendalltau(x: list, y: list, top_k: int):
    x = x[:top_k]
    y = y[:top_k]

    inter = []
    for elem in (set(x) | set(y)):
        inter.append((get_index(elem, x), get_index(elem, y)))

    ktau = 0
    n = len(inter)
    for i in range(0, n):
        for j in range(i + 1, n):
            comp1 = inter[i][0] >= inter[j][0]
            comp2 = inter[i][1] >= inter[j][1]
            if (comp1 != comp2):
                ktau += 1

    ktau = ktau / ((n * (n - 1)) / 2)

    return (1 - ktau)


def get_pos_list(rks):
    rks_pos = []
    for rk in rks:
        rk_pos = [get_index(i + 1, rk) + 1 for i, x in enumerate(rk)]
        rks_pos.append(rk_pos)
    return rks_pos


def kendall_w(rks, top_k):
    # compute pos list from ranked lists
    rks = [rk[:top_k] for rk in rks]
    rks = get_pos_list(rks)

    m = len(rks)  # number of ranked lists to compare
    n = len(rks[0])  # number of elements in each ranked list

    # compute kendall w
    r = np.sum(rks, axis=0)
    a = np.sum(r) / n
    d = [math.fabs(x - a) for x in r]
    d2 = [x**2 for x in d]
    s = np.sum(d2)
    w = (12 * s) / (m**2 * (n) * (n**2 - 1))

    # compute chi squared
    # x2 = m*(n-1)*w
    # v = chi2.isf(q=0.05, df=n-1)
    # reject = x2 < v

    return w


def compute_kendallw(x, y, top_k):
    rks = [x, y]
    return kendall_w(rks, top_k)


def compute_spearman(x, y, top_k):
    x = x[:top_k]
    y = y[:top_k]

    inter = []
    for elem in (set(x) | set(y)):
        inter.append((get_index(elem, x), get_index(elem, y)))

    spearman = 0
    n = len(inter)
    for i in range(0, n):
        spearman += abs(inter[i][0] - inter[i][1])

    spearman = spearman / (len(x) * (len(x) + 1))

    return (1 - spearman)


def compute_jaccard_k(x: list, y: list, top_k: int):
    score = 0

    x_leftover = set()
    y_leftover = set()
    stored = set()  # We only want unique values
    stored_x = set()
    stored_y = set()
    cur_inter = 0
    for i in range(top_k):
        x_elm = x[i]
        y_elm = y[i]
        if x_elm not in stored and x_elm == y_elm:
            cur_inter += 1
            stored.add(x_elm)
            stored_x.add(x_elm)
            stored_y.add(y_elm)
        else:
            if x_elm not in stored:
                if x_elm in y_leftover:
                    # x_elm was previously encountered in y
                    cur_inter += 1
                    stored.add(x_elm)
                    stored_x.add(x_elm)
                    y_leftover.remove(x_elm)
                else:
                    x_leftover.add(x_elm)
                    stored_x.add(x_elm)
            if y_elm not in stored:
                if y_elm in x_leftover:
                    # y_elf was previously encountered in x
                    cur_inter += 1
                    stored.add(y_elm)
                    stored_y.add(y_elm)
                    x_leftover.remove(y_elm)
                else:
                    y_leftover.add(y_elm)
                    stored_y.add(y_elm)

        score += cur_inter / (len(stored_x)+len(stored_y)-cur_inter)

    return score / top_k


def compute_intersection(x: list, y: list, top_k: int):
    if x[:top_k] == y[:top_k]:
        return 1

    x_leftover = set()
    y_leftover = set()
    stored = set()  # We only want unique values
    acum_inter = 0
    cur_inter = 0
    for i in range(top_k):
        x_elm = x[i]
        y_elm = y[i]
        if x_elm not in stored and x_elm == y_elm:
            cur_inter += 1
            stored.add(x_elm)
        else:
            if x_elm not in stored:
                if x_elm in y_leftover:
                    # x_elm was previously encountered in y
                    cur_inter += 1
                    stored.add(x_elm)
                    y_leftover.remove(x_elm)
                else:
                    x_leftover.add(x_elm)
            if y_elm not in stored:
                if y_elm in x_leftover:
                    # y_elf was previously encountered in a
                    cur_inter += 1
                    stored.add(y_elm)
                    x_leftover.remove(y_elm)
                else:
                    y_leftover.add(y_elm)

        acum_inter += cur_inter

    return acum_inter / ((top_k * (top_k + 1)) / 2)


def compute_rbo(x, y, top_k):
    x_leftover = set()
    y_leftover = set()
    stored = set()  # We only want unique values
    acum_inter = 0
    score = 0
    p = 0.9

    for i in range(top_k):
        x_elm = x[i]
        y_elm = y[i]
        if x_elm not in stored and x_elm == y_elm:
            acum_inter += 1
            stored.add(x_elm)
        else:
            if x_elm not in stored:
                if x_elm in y_leftover:
                    # x_elm was previously encountered in y
                    acum_inter += 1
                    stored.add(x_elm)
                    y_leftover.remove(x_elm)
                else:
                    x_leftover.add(x_elm)
            if y_elm not in stored:
                if y_elm in x_leftover:
                    # y_elf was previously encountered in x
                    acum_inter += 1
                    stored.add(y_elm)
                    x_leftover.remove(y_elm)
                else:
                    y_leftover.add(y_elm)

        score += (p**((i+1) - 1)) * (acum_inter / (i+1))

    return (1 - p) * score
