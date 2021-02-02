from numba_dppy.ocl.stubs import Stub

class numpy(Stub):
    _description_ = "<numpy>"

{% for func in numpy_supported_funcs -%}

{%     filter indent(width=4, first=True) %}
def {{ func.name }}():
{%         if func.get("nest") %}
    """This function provides numpy.{{ func.nest  }}.{{ func.name }}()
{%         else %}
    """This function provides numpy.{{ func.name }}()
{%         endif %}
    """
    pass

{%     endfilter %}
{% endfor %}
