Random Number Generation
========================

Numba-dpex does not provide a random number generation algorithm that can be
executed on the SYCL-supported Device.

Numba-dpex provides access to NumPy random algorithms that can be executed on the
SYCL-supported device via integration with `dpnp Random`_.

.. _`dpnp Random`: https://intelpython.github.io/dpnp/reference/comparison.html#random-sampling


Supported functions
-------------------

Simple random
`````````````

- `random <https://numpy.org/doc/1.16/reference/generated/numpy.random.random.html#numpy.random.random>`_
- `sample <https://numpy.org/doc/1.16/reference/generated/numpy.random.sample.html#numpy.random.sample>`_
- `ranf <https://numpy.org/doc/1.16/reference/generated/numpy.random.ranf.html#numpy.random.ranf>`_
- `random_sample <https://numpy.org/doc/1.16/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample>`_
- `rand <https://numpy.org/doc/1.16/reference/generated/numpy.random.rand.html#numpy.random.rand>`_
- `randint <https://numpy.org/doc/1.16/reference/generated/numpy.random.randint.html#numpy.random.randint>`_
- `random_integers <https://numpy.org/doc/1.16/reference/generated/numpy.random.random_integers.html#numpy.random.random_integers>`_

Distribution
````````````

- `beta <https://numpy.org/doc/1.16/reference/generated/numpy.random.beta.html#numpy.random.beta>`_
- `binomial <https://numpy.org/doc/1.16/reference/generated/numpy.random.binomial.html#numpy.random.binomial>`_
- `chisquare <https://numpy.org/doc/1.16/reference/generated/numpy.random.chisquare.html#numpy.random.chisquare>`_
- `exponential <https://numpy.org/doc/1.16/reference/generated/numpy.random.exponential.html#numpy.random.exponential>`_
- `gamma <https://numpy.org/doc/1.16/reference/generated/numpy.random.gamma.html#numpy.random.gamma>`_
- `geometric <https://numpy.org/doc/1.16/reference/generated/numpy.random.geometric.html#numpy.random.geometric>`_
- `gumbel <https://numpy.org/doc/1.16/reference/generated/numpy.random.gumbel.html#numpy.random.gumbel>`_
- `hypergeometric <https://numpy.org/doc/1.16/reference/generated/numpy.random.hypergeometric.html#numpy.random.hypergeometric>`_
- `laplace <https://numpy.org/doc/1.16/reference/generated/numpy.random.laplace.html#numpy.random.laplace>`_
- `lognormal <https://numpy.org/doc/1.16/reference/generated/numpy.random.lognormal.html#numpy.random.lognormal>`_
- `multinomial <https://numpy.org/doc/1.16/reference/generated/numpy.random.multinomial.html#numpy.random.multinomial>`_
- `multivariate_normal <https://numpy.org/doc/1.16/reference/generated/numpy.random.multivariate_normal.html#numpy.random.multivariate_normal>`_
- `negative_binomial <https://numpy.org/doc/1.16/reference/generated/numpy.random.negative_binomial.html#numpy.random.negative_binomial>`_
- `normal <https://numpy.org/doc/1.16/reference/generated/numpy.random.normal.html#numpy.random.normal>`_
- `poisson <https://numpy.org/doc/1.16/reference/generated/numpy.random.poisson.html#numpy.random.poisson>`_
- `rayleigh <https://numpy.org/doc/1.16/reference/generated/numpy.random.rayleigh.html#numpy.random.rayleigh>`_
- `standard_cauchy <https://numpy.org/doc/1.16/reference/generated/numpy.random.standard_cauchy.html#numpy.random.standard_cauchy>`_
- `standard_exponential <https://numpy.org/doc/1.16/reference/generated/numpy.random.standard_exponential.html#numpy.random.standard_exponential>`_
- `standard_gamma <https://numpy.org/doc/1.16/reference/generated/numpy.random.standard_gamma.html#numpy.random.standard_gamma>`_
- `standard_normal <https://numpy.org/doc/1.16/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal>`_
- `uniform <https://numpy.org/doc/1.16/reference/generated/numpy.random.uniform.html#numpy.random.uniform>`_
- `weibull <https://numpy.org/doc/1.16/reference/generated/numpy.random.weibull.html#numpy.random.weibull>`_

Example:

.. note::
    To ensure the code is executed on GPU set ``DEBUG=1`` (or ``NUMBA_DPEX_DEBUG=1``) and look to stdout

.. literalinclude:: ./../../../../numba_dpex/examples/rand.py
