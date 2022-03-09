Sample Space
============


|pypi| |py_versions| |license| |doi|
|rtfd| |codecov| |github-actions|

.. |pypi| image:: https://img.shields.io/pypi/v/sspace.svg
    :target: https://pypi.python.org/pypi/sspace
    :alt: Current PyPi Version

.. |py_versions| image:: https://img.shields.io/pypi/pyversions/sspace.svg
    :target: https://pypi.python.org/pypi/sspace
    :alt: Supported Python Versions

.. |license| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: BSD 3-clause license

.. |rtfd| image:: https://readthedocs.org/projects/sample-space/badge/?version=latest
    :target: https://sample-space.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |codecov| image:: https://codecov.io/gh/Epistimio/sample-space/branch/master/graph/badge.svg?token=4uVIfUhbfb
    :target: https://codecov.io/gh/Epistimio/sample-space
    :alt: Codecov Report

.. |github-actions| image::https://github.com/Epistimio/sample-space/actions/workflows/test.yml/badge.svg?branch=master&event=pull_request
    :target: https://github.com/Epistimio/sample-space/actions/workflows/test.yml
    :alt: Github actions tests


Unified search space definition library


Features
--------

* Support ConfigSpace dimensions
* Support Scipy dimensions


User friendly API
----------------

.. code-block:: python

    from sspace import Space, either, eq
    import json

    space = Space(backend='ConfigSpace')

    # Add a new dimensions to the search space
    # optim is a reference to the created dimension
    optim = space.categorical('optimizer', ['sgd', 'adam'])

    # Add another dimension
    optim_lr = space.loguniform('optimizer.lr', 1, 2, quantization=0.01)

    # dimensions can have constraints
    optim_lr.forbid_equal(1)

    # You can use the reference to add constraint
    # here learning-rate is only available if the optimizer is adam or sgd
    optim_lr.enable_if(either(eq(optim, 'adam'), eq(optim, 'sgd')))


    for sample in space.sample(2):
        print(sample)

    print(json.dumps(space.serialize(), indent=2))


Seamless Integration
--------------------

With hyperparameter decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> @hyperparameter(a=uniform(0, 1), b=normal(0, 1))
... def fun(a, b):
...    return a + b

>>> get_space(fun).serialize()
{'a': 'uniform(lower=0, upper=1)', 'b': 'normal(loc=0, scale=1)'}


With hyperparameter annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> def fun(a: ordinal(1, 2, 3)):
...    return a

>>> get_space(fun).serialize()
{'a': 'ordinal(sequence=[1, 2, 3])'}
