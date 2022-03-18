#   -*- coding: utf-8 -*-
from pybuilder.core import use_plugin, init, Author

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")


name = "NNGUI"
version = "0.1"
url = "https://github.com/Qessia/NNGUI"
description = """Simple desktop application for comfortable working with your neural network"""
authors = [Author("Timur Gayazov", "timurgayazovfw@gmail.com", "application development"),
           Author("Vladimir Taratutin", "v.taraturin@g.nsu.ru", "NN interfaces"),
           Author("Mikhail Klementiev", "m.klementiev@g.nsu.ru", "design & documentation")]
license = "None"
default_task = "publish"


@init
def set_properties(project):
    pass
