from setuptools import setup
import sys

#abort installation if Python version does not meet requirement
if sys.version_info < (3, 5):
      sys.exit('Sorry, Python < 3.5 is not supported. Please update to Python 3.5 or above')

setup(
      name              = 'calipy',
      packages          = ['calipy'],
      version           = '0.1',
      description       = 'Devices Calibration Toolbox',
      author            = 'Jaedong kim',
      author_email      = 'jaedong27@gmail.com',
      url               = 'https://github.com/jaedong27/calipy',
      license           = 'BSD',
      keywords          = ['Camera Calibration', 'Projector Calibration'],
      python_requires   = '>=3',
      install_requires  = [
                              'numpy'
                        ]
)