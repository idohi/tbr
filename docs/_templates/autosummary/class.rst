{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{% if fullname.startswith("tbr.core.results.") %}
.. autoclass:: {{ objname }}()
   :members:
   :show-inheritance:
   :class-doc-from: class
{% else %}
.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
{% endif %}
