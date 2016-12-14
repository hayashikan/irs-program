# -*- coding: utf-8 -*-
"""
Project: MAM Integrated Reporting System
Author: LIN, Han (Jo)
"""
# import modules
import os
import sys
import inspect

# import modules in subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "resources")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

# import IRS in subfolder "resources"
from IRS import Integrated_Reporting_System

# DO NOT MODIFY CODE BEFORE HERE -----------------------------------------

# run report by following code -------------------------------------------
# 'default.mamspec' file is in the same folder as this program
# you can change the file name as the .mamspec file in this folder
IRS = Integrated_Reporting_System('default.mamspec')
IRS.generate_report() # this command is to generate report
