from glob import glob

from setuptools import setup

PACKAGE = "crane_planning"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[PACKAGE],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    # colcon picks its pytest step off `tests_require`, not off package.xml, so
    # without this line `test/` is never run.
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Architecture maintainers",
    maintainer_email="maintainers@example.invalid",
    description="Deterministic collision-free crane corridors and shaped timing.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            # The name `crane_bringup`'s profiles compose, unchanged from when
            # this package was C++: it is a cross-node contract, not a detail.
            f"crane_planner_node = {PACKAGE}.node:main",
        ],
    },
)
