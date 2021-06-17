# Copyright 2020, 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numba import njit


@njit(debug=True)
def foo(arg):
    l1 = arg + 6
    l2 = arg * 5.43
    l3 = (arg, l1, l2, "bar")
    print(arg, l1, l2, l3)


def main():
    result = foo(987)
    print(result)


if __name__ == '__main__':
    main()
