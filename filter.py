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
RESOLUTION = 128, 128


def main():

    dvs = SimDvs(threshold=4, resolution=RESOLUTION)

    cap = cv2.VideoCapture(0)

    noise_filter = SpatioTemporalCorrelationFilter()

    start = time()

    nofiltimg = np.zeros(RESOLUTION)

    while cap.isOpened():

        _, image = cap.read()

        events = dvs.getEvents(image)

        filtered = (dvs.filter(events, noise_filter)
                    if (np.count_nonzero(events) / np.prod(events.shape) <
                        DENSITY_THRESHOLD)
                    else nofiltimg)

        # if not dvs.display(image, events, filtered, scaleup=2):
        if not dvs.display(image, events, filtered=filtered, scaleup=2):

            break


main()
