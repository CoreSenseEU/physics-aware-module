import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'physics_amm'


def ansr_source_data_files():
    """Install the vendored ANSR core into share/physics_amm/ansr_source/.

    Only code and package metadata are installed — the bundled benchmark
    datasets under source_code/data/ are large and not needed at runtime
    (sessions stage their own data).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, '..', '..', 'source_code')
    out = []
    for pattern in ('*.py', 'requirements.txt', 'constraints/**/*.py'):
        for path in glob(os.path.join(src, pattern), recursive=True):
            rel_dir = os.path.dirname(os.path.relpath(path, src))
            dest = os.path.join('share', package_name, 'ansr_source', rel_dir)
            out.append((os.path.normpath(dest),
                        [os.path.relpath(path, here)]))
    return out


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ] + ansr_source_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Erik Derner',
    maintainer_email='erik.derner@aiglu.cz',
    description='ROS 2 wrapper for the PhysicsAMM / ANSR symbolic-regression module.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'physics_amm_node = physics_amm.node:main',
        ],
    },
)
