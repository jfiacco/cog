COG_PATH_PREFIX = "/tmp"
COG_ROOT = "cog_root"
COG_HOME = "cog-test"
COG_SYS_DIR = "sys"
COG_SYS_FILE = "cogsys.c"
COG_DEFAULT_NAMESPACE = "default"
VIEWS = 'views'
STORE="-store-"
INDEX="-index-"
INDEX_BLOCK_LEN = 32
INDEX_CAPACITY = 100003 # must be a prime number
STORE_READ_BUFFER_SIZE = 512
LEVEL_2_CACHE_SIZE = 100000

''' TORQUE '''
GRAPH_NODE_SET_TABLE_NAME = 'TOR_NODE_SET'
GRAPH_EDGE_SET_TABLE_NAME = 'TOR_EDGE_SET'
EMBEDDING_SET_TABLE_NAME = 'EMBEDDING_SET'

''' HNSW INDEX CONFIGURATION '''
# HNSW index type configuration
HNSW_ENABLED = False  # Enable HNSW indexing for embeddings by default
HNSW_SPACE = 'cosine'  # Distance metric: 'cosine', 'l2', or 'ip' (inner product)
HNSW_EF_CONSTRUCTION = 200  # Controls index construction speed/accuracy tradeoff
HNSW_EF = 50  # Controls query time/accuracy tradeoff
HNSW_M = 16  # Number of bi-directional links per element
HNSW_MAX_ELEMENTS = 10000  # Maximum number of elements in the index
HNSW_ALLOW_REPLACE_DELETED = False  # Allow replacing deleted elements
HNSW_NUM_THREADS = 1  # Number of threads for index operations
HNSW_INDEX_DIR = 'hnsw_indices'  # Directory name for HNSW indices

''' CUSTOM COG DB PATH '''
CUSTOM_COG_DB_PATH = None


def cog_db_path():
    if CUSTOM_COG_DB_PATH:
        return CUSTOM_COG_DB_PATH
    else:
        return "/".join([COG_PATH_PREFIX, COG_HOME])


def cog_context():
    return [cog_db_path(), COG_SYS_DIR, COG_SYS_FILE]


def cog_instance_sys_file():
    return "/".join(cog_context())


def cog_instance_sys_dir():
    return "/".join(cog_context()[0:-1])


def cog_views_dir():
    return "/".join([cog_db_path(), VIEWS])


def cog_hnsw_dir():
    return "/".join([cog_db_path(), HNSW_INDEX_DIR])


def cog_data_dir(db_name):
    return "/".join(cog_context()[0:-2]+[db_name])


def cog_index(db_name, table_name, instance_id, index_id):
    return "/".join(cog_context()[0:-2]+[db_name,table_name+INDEX+instance_id+"-"+str(index_id)])


def get_table_name(index_file_name):
    return index_file_name.split(INDEX)[0]


def index_id(index_name):
    return int(index_name.split("-")[-1])


def cog_store(db_name, table_name, instance_id):
    return "/".join(cog_context()[0:-2]+[db_name, table_name+STORE+instance_id])


import logging

logging_config = dict(
    version = 1,
    formatters = {
        'f': {'format':
              '%(asctime)s %(name)-12s %(levelname)-8s [%(filename)s:%(lineno)s - %(funcName)10s()] %(message)s'}
        },
    handlers = {
        'h': {'class': 'logging.StreamHandler',
              'formatter': 'f'
              }
        },
    root = {
        'handlers': ['h'],
        'level': logging.WARN,
        }
)