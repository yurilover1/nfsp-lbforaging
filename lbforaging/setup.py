from setuptools import setup, find_packages

setup(
    name="lbforaging",
    version="4.0.0",
    description="Level Based Foraging Environment",
    author="yurilover1",
    url="https://github.com/yurilover1/nfsp-lbforaging",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11"
    ],
    install_requires=["numpy", "gymnasium", "six"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
