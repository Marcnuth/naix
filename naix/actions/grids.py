import random
import numpy as np
from itertools import product


def minimal_actions(seed, shape, stride, drag_len):
    targets = [(x + stride[0] / 2, y + stride[1] / 2) for x in range(0, shape[0], stride[0]) for y in range(0, shape[1], stride[1])]

    clicks = list(product(['click'], targets))
    drag_ups = list(_drag_actions([(np.array(shape) / 2).tolist()], [0, -1 * drag_len])) # only one drag up, avoid search complex
    actions = clicks + drag_ups

    random.seed(seed)
    random.shuffle(actions)
    return actions


def common_actions(seed, shape, stride, drag_len):
    targets = [(x + stride[0] / 2, y + stride[1] / 2) for x in range(0, shape[0], stride[0]) for y in
               range(0, shape[1], stride[1])]

    clicks = list(product(['click'], targets))
    drag_ups = list(_drag_actions(targets, [0, -1 * drag_len]))
    drag_downs = list(_drag_actions(targets, [0, 1 * drag_len]))
    drag_lefts = list(_drag_actions(targets, [-1 * drag_len, 0]))
    drag_rights = list(_drag_actions(targets, [1 * drag_len, 0]))
    inputs = [('input',)]
    kevents = [('delete',)]
    actions = clicks + drag_ups + drag_downs + drag_lefts + drag_rights + inputs + kevents

    random.seed(seed)
    random.shuffle(actions)
    return actions


def _drag_actions(targets, direction):
    ctx = np.concatenate((np.array(targets), np.array(targets) + np.array(direction)), axis=1)
    conditions = np.any(ctx < 0, axis=1) | (ctx[:, 0] > 1080) | (ctx[:, 1] > 1920) | (ctx[:, 2] > 1080) | (
    ctx[:, 3] > 1920)
    ctx = np.delete(ctx, np.argwhere(conditions), axis=0)
    yield from product(['drag'], ctx.tolist())