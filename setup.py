from setuptools import setup, find_packages
import grandma_stock_valuation


DISTNAME = "grandma-stock-valuation"
DESCRIPTION = "A simple, manageable valuation tool and portfolio builder for retail investors."
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Yang Xi"
URL = "https://github.com/gowestyang/grandma_stock_valuation"
LICENSE = "MIT"
VERSION = grandma_stock_valuation.__version__


def setup_package():
    metadata = dict(
        name=DISTNAME,
        version=VERSION,
        maintainer=MAINTAINER,
        license=LICENSE,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,

        classifiers=[
                "Intended Audience :: Science/Research",
                "Intended Audience :: Developers",
                "License :: MIT",
                "Programming Language :: Python",
                "Topic :: Software Development",
                "Topic :: Scientific/Engineering",
                "Development Status :: 5 - Production/Stable",
                "Operating System :: Microsoft :: Windows",
                "Operating System :: POSIX",
                "Operating System :: Unix",
                "Operating System :: MacOS",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10"
            ],

        python_requires=">=3.7",
        install_requires=["numpy", "pandas", "scikit-learn", "plotly"],

        packages=find_packages('grandma_stock_valuation'),
        package_dir={'': 'grandma_stock_valuation'},
        #keywords='example project',
        #package_data={"": ["*.pxd"]}
    )

    setup(**metadata)

if __name__ == "__main__":
    setup_package()
