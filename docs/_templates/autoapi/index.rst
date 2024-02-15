API Reference
=============

This page contains auto-generated API reference documentation [#f1]_.

.. toctree::
   :maxdepth: 2

   numba_dpex/kernel_api/index

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
