# Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
#
# This file is part of DREAM ML.
#
# DREAM ML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DREAM ML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DREAM ML. If not, see <https://www.gnu.org/licenses/>.

# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("api.views", ["api/views.py"]),
    Extension("api.train", ["api/train.py"]),
    Extension("api.data_cleaning", ["api/data_cleaning.py"]),
    Extension("api.data_encoding", ["api/data_encoding.py"]),
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"}
    ),
)