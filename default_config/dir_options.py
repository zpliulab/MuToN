def dir_opts(base_dir='data/SKEMPI'):
    dir_opts = {}
    dir_opts['base_dir'] = base_dir
    dir_opts['complex_dir'] = '{}/raws/'.format(base_dir)
    dir_opts['single_dir'] = '{}/raw_pdb/'.format(base_dir)
    dir_opts['llm_dir'] = '{}/llm_embedding/'.format(base_dir)
    dir_opts['numpy_dir'] = 'data/SKEMPI/numpy_dir/'
    return dir_opts