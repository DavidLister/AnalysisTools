# common.py
#
# A file to hold shared symbolic constants.
# David Lister
# October 2023
#

import pint

ureg = pint.UnitRegistry()

FIXED_PARAMETERS = "FIXED_PARAMETERS"
FIT_PARAMETERS = "FIT_PARAMETERS"
PARAMETERS = "PARAMETERS"
MODEL = "MODEL"
DOMAIN = "DOMAIN"

AUTO = "AUTO"  # For automatically determining initial guess

DEBUG = False
