#!/usr/bin/env python3
#
#  Copyright (C) 2023 Simon D. Levy
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from distutils.core import setup

setup(
        name='simdvs',
        version='0.1',
        description='Simple Dynamic Vision Sensor simulator in Python',
        author='Simon D. Levy',
        author_email='simon.d.levy@gmail.com',
        url='https://github.com/simondlevy/simdvs',
        packages=['simdvs'],
        install_requires=['opencv-python']
        )
