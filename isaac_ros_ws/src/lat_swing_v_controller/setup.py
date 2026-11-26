from setuptools import find_packages, setup

package_name = 'lat_swing_v_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orin64',
    maintainer_email='santiago@ai.iit.tsukuba.ac.jp',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'feedforward_lat_swing = lat_swing_v_controller.feedforward_lat_swing:main'
        ],
    },
)
