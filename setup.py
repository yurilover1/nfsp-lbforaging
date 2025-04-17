from setuptools import setup, find_packages

setup(
    name="lbforaging",
    version="3.0.0",
    description="Level Based Foraging Environment",
    author="yurilover1",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numpy", "gymnasium", "pyglet<2", "six"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
