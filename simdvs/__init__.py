'''
A Python class to simulate a Dynamic Vision Sensor through first-differencing
of current and previous image from an ordinary camera.

Copyright (C) 2023 Simon D. Levy, Armando Mendez-Anastasio

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

import numpy as np
import cv2


class SimDvs:

    def __init__( self, resolution=None, threshold=0):

        self.resolution = resolution
        self.threshold = threshold

        self.image_prev = None

    def getEvents(self, image):
        '''
        Returns current event image, or None if user quits display
        '''

        # Convert the full-size image to grayscale
        graycurr = self._color2gray(image)

        # Create an empty events array of the same size as the grayscale image
        events = np.zeros(graycurr.shape, dtype=np.int8)

        # Once we have a current and previous image, we can compute events
        if self.image_prev is not None:

            # Make a first-difference image between the current and previous
            # grayscale images
            grayprev = self._color2gray(self.image_prev)
            diffimg = graycurr - grayprev

            # Simulate positive and negative events w.r.t. a threshold
            events[diffimg > +self.threshold] = +1
            events[diffimg < -self.threshold] = -1

        # Track the previous image for first-differencing
        self.image_prev = image

        # If sensor resolution was indicated, resize the event image now
        if self.resolution is not None:
            events = cv2.resize(events.astype('float32'), (128, 128))

        return events

    def annotate(self, ceventimg):
        '''
        Override this method to annotate the event image.
        '''

        pass

    def display(
            self, image, events, 
            filtered=None, scaleup=1, quit_key=27, colorize=True):

        # Make a color image from the event image
        rows, cols = events.shape
        ceventimg = self._colorize(events, colorize)

        # Support annotating the event image in a subclass
        self.annotate(ceventimg)

        # Make two-column image to display the original and events, or three columns
        k = 2 if filtered is None else 3
        rows, cols = events.shape
        bigimg = np.zeros((rows, k * cols, 3)).astype(np.uint8)

        # Fill the first column with the original (resized if # indicated)
        bigimg[:, :cols, :] = (
                image if self.resolution is None 
                else cv2.resize(image, self.resolution))

        # Fill the second column with the events image
        bigimg[:, cols:(2*cols), :] = ceventimg

        # Fill the third column with the filtered image if provided
        if filtered is not None:
            cfiltered = self._colorize(filtered, colorize)
            self.annotate(cfiltered)
            bigimg[:, (2*cols):(3*cols), :] = cfiltered

        # Display the big image
        cv2.imshow('Events',
                   cv2.resize(bigimg,
                              (scaleup * bigimg.shape[1],
                               scaleup * bigimg.shape[0])))

        # Check whether the user hit the quit key
        if cv2.waitKey(1) == quit_key:
            return False

        return True

    def _colorize(self, events, colorize):

        rows, cols = events.shape
        ceventimg = np.zeros((rows, cols, 3))

        # If color was indicated, display positive events as green,
        # negative as red
        if colorize:

            ceventimg[events == +1, 1] = 255
            ceventimg[events == -1, 2] = 255

        # Otherwise, display all events as white
        else:
            ceventimg[events != 0, :] = 255

        return ceventimg

    def _color2gray(self, img):

        # Convert to grayscale and downsample for subtraction
        return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) >> 1).astype(np.int8)
