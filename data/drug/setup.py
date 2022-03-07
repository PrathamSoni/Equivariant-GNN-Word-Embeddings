import setuptools

setuptools.setup(
    name="machinereading",
    packages=setuptools.find_packages(),
    package_dir={"machinereading": "machinereading"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)