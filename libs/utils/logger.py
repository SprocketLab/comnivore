import os
import shutil
from .graph_modules import show_graph


def set_log_path(path):
    global _log_path
    _log_path = path
    if os.path.isfile(os.path.join(path,'log.txt')):
        os.remove(os.path.join(path,'log.txt'))

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def save_graph(graph, title):
    show_graph(graph, size=(2,2),  title=title, show=False, figname=os.path.join(_log_path, f"{title}.png"), save=True)