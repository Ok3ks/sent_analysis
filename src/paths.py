from os.path import realpath, join, dirname

SRC_DIR = realpath(dirname(__file__))
ROOT_DIR = realpath(join(SRC_DIR, '..'))
DATA_DIR = realpath(join(ROOT_DIR, 'data'))
MODEL_DIR = realpath(join(ROOT_DIR, 'models'))
LOG_DIR = realpath(join(ROOT_DIR, 'logs'))
CHUNK_DIR = realpath(join(DATA_DIR, 'chunk'))
CONFIG_DIR = realpath(join(ROOT_DIR, 'configs'))