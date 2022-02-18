from setuptools import setup
from os import path
import grandma_stock_valuation

DISTNAME = "grandma-stock-valuation"
DESCRIPTION = "A simple, manageable valuation tool and portfolio builder for retail investors."
with open("DESCRIPTION.md") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = "Yang Xi"
URL = "https://github.com/gowestyang/grandma-stock-valuation"
LICENSE = "MIT"
VERSION = grandma_stock_valuation.__version__


def setup_package():
    metadata = dict(
        name=DISTNAME,
        version=VERSION,
        packages=["grandma_stock_valuation", path.join("grandma_stock_valuation", "utils")],
        include_package_data=True,

        author=AUTHOR,
        license=LICENSE,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        classifiers=[
                "Development Status :: 3 - Alpha",
                "License :: OSI Approved :: MIT License",
                "Intended Audience :: Science/Research",
                "Intended Audience :: Developers",
                "Topic :: Scientific/Engineering",
                "Topic :: Office/Business :: Financial :: Investment",

                "Programming Language :: Python",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11"
            ],

        python_requires=">=3.7",
        install_requires=["numpy", "pandas", "scikit-learn", "plotly"]
    )

    setup(**metadata)

if __name__ == "__main__":
    setup_package()
