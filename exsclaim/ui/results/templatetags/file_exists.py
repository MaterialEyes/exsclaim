from django import template
import pathlib
from exsclaim.ui.exsclaim_gui.settings import STATICFILES_DIRS

register = template.Library()

@register.filter
def exists(value):
    """ returns True if files exists in any STATICFILES_DIRS """
    for dir in STATICFILES_DIRS:
        potential_path = dir / value
        if potential_path.exists():
            return True
    return False
