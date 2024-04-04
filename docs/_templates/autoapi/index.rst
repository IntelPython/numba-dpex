API Reference
=============

This page contains auto-generated API reference documentation [#f1]_.

.. toctree::
   :maxdepth: 1

   numba_dpex/kernel_api/index
   numba_dpex/core/decorators/index
   numba_dpex/core/kernel_launcher/index

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
