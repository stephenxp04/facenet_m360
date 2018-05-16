from google.appengine.ext import vendor
import os

vendor.add('lib')
#if os.environ.get('SERVER_SOFTWARE', '').startswith('Development'):
#    import imp
#    import os.path
#    import inspect
#    from google.appengine.tools.devappserver2.python import sandbox
#
#    sandbox._WHITE_LIST_C_MODULES += ['_ssl', '_socket']
    # Use the system socket.

#    real_os_src_path = os.path.realpath(inspect.getsourcefile(os))
#    psocket = os.path.join(os.path.dirname(real_os_src_path), 'socket.py')
#    imp.load_source('socket', psocket)