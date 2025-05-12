#!/usr/bin/env python

from setuptools import setup

setup(name='mpl_plot_templates',
      version='0.1.0',
      description='Template plots for matplotlib',
      author='Adam Ginsburg',
      author_email='adam.g.ginsburg@gmail.com',
      packages=['mpl_plot_templates'],
      provides=['mpl_plot_templates'],
      install_requires=[
          'matplotlib',
          'opencv-python>=4.5.0',
          'numpy>=1.19.0',
          'Pillow>=8.0.0',
      ],
      keywords=['Scientific/Engineering'],
      )
