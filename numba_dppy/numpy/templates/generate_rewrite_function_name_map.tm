import numpy as np

rewrite_function_name_map = {
{% for func in numpy_supported_funcs -%}
{%     if func.impl == "dpnp" %}
{%         filter indent(width=4, first=True) %}
{%             if func.get("nest") %}
"{{ func.name }}" : {"{{ func.nest }}" : (["numpy"], "{{ func.name }}")},
{%             else %}
"{{ func.name }}" : (["numpy"], "{{ func.name }}"),
{%             endif %}
{%         endfilter %}
{%     endif %}
{% endfor %}
}

