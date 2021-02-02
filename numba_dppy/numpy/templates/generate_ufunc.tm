numba_dppy_numpy_ufunc = [
{% for func in numpy_supported_funcs -%}
{%     if func.impl == "ufunc" %}
{%         filter indent(width=4, first=True) %}
{%             if func.get("alt") %}
("{{ func.alt }}", np.{{ func.name }}),
{%             else %}
("{{ func.name }}", np.{{ func.name }}),
{%             endif %}
{%         endfilter %}
{%     endif %}
{% endfor %}
]

