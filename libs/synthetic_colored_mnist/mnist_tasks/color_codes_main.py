from .color_codes import RGB
from collections import namedtuple, OrderedDict

RED = RGB(255,0,0)
GREEN = RGB(0,255,0)
BLUE = RGB(0,0,254)
YELLOW = RGB(255,255,0)
BROWN = RGB(150,75,0)
GRAY = RGB(128,128,128)
VIOLET = RGB(134,1,175)
BLACK = RGB(0,0,0)
WHITE = RGB(1, 1, 1)
ORANGE = RGB(255,128,0)
TEAL = RGB(0,128,128)	
PINK = RGB(255,192,203)	

colors = {} #dict of colors
colors['red'] = RED
colors['green'] = GREEN
colors['yellow'] = YELLOW
colors['brown'] = BROWN
colors['blue'] = BLUE
colors['gray'] = GRAY
colors['violet'] = VIOLET
# colors['black'] = BLACK
colors['white'] = WHITE
colors['orange'] = ORANGE
colors['teal'] = TEAL
colors['pink'] = PINK

colors = OrderedDict(sorted(colors.items(), key=lambda t: t[0]))