from .color_codes import RGB
from collections import namedtuple, OrderedDict

RED = RGB(255,0,0)
GREEN = RGB(0,255,0)
YELLOW = RGB(255,255,0)
BROWN = RGB(150,75,0)
BLUE = RGB(0,0,254)
GRAY = RGB(128,128,128)
VIOLET = RGB(134,1,175)

colors = {} #dict of colors
colors['red'] = RED
colors['green'] = GREEN
colors['yellow'] = YELLOW
colors['brown'] = BROWN
colors['blue'] = BLUE
colors['gray'] = GRAY
colors['violet'] = VIOLET

colors = OrderedDict(sorted(colors.items(), key=lambda t: t[0]))