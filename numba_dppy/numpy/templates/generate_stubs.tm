from numba_dppy.ocl.stubs import Stub

class numpy(Stub):
    _description_ = "<numpy>"

{% for func in numpy_supported_funcs -%}

{%     filter indent(width=4, first=True) %}
def {{ func.name }}():
    pass

{%     endfilter %}
{% endfor %}
