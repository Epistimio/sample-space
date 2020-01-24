import pytest
import json

from sspace import Space, either, both, eq, ne, lt, gt, contains
from sspace.convert import convert_space

backends = ['ConfigSpace', 'Orion']
conditions = [eq, ne, lt, gt]

orion_space = {
    'unf': 'uniform(0, 1)',
    'uni': 'uniform(0, 1, discrete=True)',
    'cat': 'choices(["a", "b"])',
    'cac': 'choices({"a": 0.2, "b": 0.8})',
    'fid': 'fidelity(1, 300, 4)',
    'lun': 'loguniform(1, 2)',
    'nor': 'normal(0, 1)',
    'gau': 'gaussian(1, 1)',
}


def make_space(backend='ConfigSapce'):
    space = Space(backend=backend)
    optim = space.categorical('optimizer', ['sgd', 'adam'])
    sgd_lr = space.loguniform('optimizer.lr', 1, 2, quantization=0.01)
    sgd_lr.enable_if(either(eq(optim, 'adam'), eq(optim, 'sgd')))
    sgd_lr.forbid_equal(1)
    return space


@pytest.mark.parametrize('backend', backends)
def test_space_explicit(backend):
    space = make_space()
    space.instantiate(backend)
    space.sample(2)
    json.dumps(space.serialize(), indent=2)


@pytest.mark.parametrize('backend', backends)
def test_space_implicit(backend):
    space = make_space(backend)
    space.sample(2)
    json.dumps(space.serialize(), indent=2)


@pytest.mark.parametrize('backend', backends)
def test_serialization_is_same(backend):
    import copy

    space = make_space(backend)
    serialized = space.serialize()

    new_space = Space.from_dict(copy.deepcopy(serialized))
    new_serialized = new_space.serialize()
    assert serialized == new_serialized

    new_space.sample()


@pytest.mark.parametrize('backend', backends)
def test_normal(backend):
    for discrete in [True, False]:
        for log in [True, False]:
            for q in [None, 0.01, 1]:
                space = Space(backend=backend)

                space.normal(f'a_{discrete}_{log}_{q}',
                             loc=1, scale=2,
                             discrete=discrete,
                             log=log,
                             quantization=q)

                try:
                    print(space.sample())
                except NotImplementedError:
                    assert backend == 'Orion' and log is True


@pytest.mark.parametrize('backend', backends)
def test_uniform(backend):
    space = Space(backend=backend)

    for discrete in [True, False]:
        for log in [True, False]:
            for q in [None, 0.01, 1]:
                space.uniform(f'a_{discrete}_{log}_{q}', 1, 2,
                              discrete=discrete,
                              log=log,
                              quantization=q)

    print(space.sample())


@pytest.mark.parametrize('backend', backends)
def test_categorical(backend):
    space = Space(backend=backend)

    space.categorical('cat', ['a', 'b', 'c'])
    space.categorical('caw', a=0.2, b=0.1, c=0.7)
    space.categorical('cad', dict(a=0.2, b=0.1, c=0.7))

    print(space.sample())


@pytest.mark.parametrize('backend', backends)
def test_ordinal(backend):
    space = Space(backend=backend)

    try:
        space.ordinal('ord', ['a', 'b', 'c'])

        print(space.sample())
        print(space.sample())
        print(space.sample())
    except NotImplementedError:
        assert backend == 'Orion'


@pytest.mark.parametrize('backend', backends)
def test_subspace(backend):
    space = Space(backend=backend)

    space.normal('a', 1, 2, quantization=0.01)
    subspace = space.subspace('b')
    subspace.normal('a', 1, 2, quantization=0.01)

    print(space.sample())


@pytest.mark.parametrize('condition', conditions)
def test_conditions(condition):
    space = Space('ConfigSpace')

    a = space.normal('a', 1, 2, quantization=0.01)
    b = space.normal('b', 1, 2, quantization=0.01)
    b.enable_if(condition(a, 1.5))
    print(space.sample())


def test_conditions_in():
    space = Space('ConfigSpace')

    a = space.normal('a', 1, 2, quantization=0.01)
    b = space.normal('b', 1, 2, quantization=0.01)
    b.enable_if(contains(a, [1, 1.5, 2]))
    print(space.sample())


def test_conditions_and():
    space = Space('ConfigSpace')

    a = space.normal('a', 1, 2, quantization=0.01)
    b = space.normal('b', 1, 2, quantization=0.01)
    b.enable_if(both(gt(a, 1), lt(a, 2)))
    print(space.sample())


def test_conditions_or():
    space = Space('ConfigSpace')

    a = space.normal('a', 1, 2, quantization=0.01)
    b = space.normal('b', 1, 2, quantization=0.01)
    b.enable_if(either(eq(a, 1), ne(a, 2)))
    print(space.sample())


def test_forbid_eq():
    space = Space('ConfigSpace')

    a = space.uniform('a', 1, 2, quantization=0.01)
    a.forbid_equal(1)
    print(space.sample())


def test_forbid_contains():
    space = Space('ConfigSpace')

    a = space.uniform('a', 1, 2, quantization=0.01)
    a.forbid_in([1, 2])
    print(space.sample())


def test_forbid_and():
    space = Space('ConfigSpace')

    a = space.uniform('a', 1, 2, quantization=0.01)
    a.forbid_equal(1)
    a.forbid_in([1, 2])
    print(space.sample())


def test_conversion():
    cs = convert_space(orion_space)
    print(cs)
    cs.sample_configuration()


def test_deserialize_orion():
    cs = Space.from_dict(orion_space)
    print(cs)
    print(cs.sample())
    print(json.dumps(cs.serialize(), indent=2))


if __name__ == '__main__':
    test_deserialize_orion()
