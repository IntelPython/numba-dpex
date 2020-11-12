from setuptools import find_packages, setup

packages = find_packages(include=["numba_dppy", "numba_dppy.*"])

install_requires = [
    'numba',
    'dpctl',
]

metadata = dict(
    name="numba-dppy",
    description="Numba extension for Intel CPU and GPU backend",
    packages=packages,
    install_requires=install_requires,
)

setup(**metadata)

