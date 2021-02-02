numba_dppy_numpy_ufunc = [
{% for func in numpy_supported_funcs -%}
{%     if func.impl == "ufunc" %}
{%         filter indent(width=4, first=True) %}
("{{ func.name }}", np.{{ func.name }}),
{%         endfilter %}
{%     endif %}
{% endfor %}
]

