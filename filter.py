#!/usr/bin/python3
'''
Demo simdvs with camera and noise filtering

Copyright (C) 2023 Simon D. Levy

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 51
Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

import cv2
import numpy as np
from time import time

from simdvs import SimDvs

from dvs_filters.stcf import SpatioTemporalCorrelationFilter

DENSITY_THRESHOLD = 0.01


class Event:

    def __init__(self, timestamp, x, y):

        self.timestamp = timestamp
        self.x = x
        self.y = y


def filter_noise(events, noise_filter, start_time):

    filtered = np.zeros(events.shape)

    nz = np.where(events)

    for x, y in zip(nz[0], nz[1]):

        e = Event(int((time() - start_time) * 1e6), x, y)

        if noise_filter.check(e):

            filtered[x, y] = events[x, y]

    return filtered


def main():

    dvs = SimDvs(threshold=4, resolution=(128, 128))

    cap = cv2.VideoCapture(0)

    noise_filter = SpatioTemporalCorrelationFilter()

    start = time()

    while cap.isOpened():

        _, image = cap.read()

        events = dvs.getEvents(image)

        filtered = (filter_noise(events, noise_filter, start)
                    if (np.count_nonzero(events) / np.prod(events.shape) <
                        DENSITY_THRESHOLD)
                    else None)

        if not dvs.display(image, events, filtered, scaleup=2):

            break


main()
