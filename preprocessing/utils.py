"""
Utility Functions
"""

def error_handle_wrapper(function, error_return = None):

    def wrap(*args, **kwargs):

        try:
            x = function(*args,**kwargs)
        except:
            return error_return
        else:
            return x
        
    return wrap