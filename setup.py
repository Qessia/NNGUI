#!/usr/bin/env python
#   -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.command.install import install as _install

class install(_install):
    def pre_install_script(self):
        pass

    def post_install_script(self):
        pass

    def run(self):
        self.pre_install_script()

        _install.run(self)

        self.post_install_script()

if __name__ == '__main__':
    setup(
        name = 'NNGUI',
        version = '0.1',
        description = '',
        long_description = 'Simple desktop application for comfortable working with your neural network',
        long_description_content_type = None,
        classifiers = [
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python'
        ],
        keywords = '',

        author = 'Timur Gayazov, Vladimir Taratutin, Mikhail Klementiev',
        author_email = 'timurgayazovfw@gmail.com, v.taraturin@g.nsu.ru, m.klementiev@g.nsu.ru',
        maintainer = '',
        maintainer_email = '',

        license = 'None',

        url = 'https://github.com/Qessia/NNGUI',
        project_urls = {},

        scripts = [],
        packages = [],
        namespace_packages = [],
        py_modules = [
            'features-testing',
            'main',
            'nntemplate',
            'test'
        ],
        entry_points = {},
        data_files = [],
        package_data = {},
        install_requires = [],
        dependency_links = [],
        zip_safe = True,
        cmdclass = {'install': install},
        python_requires = '',
        obsoletes = [],
    )
