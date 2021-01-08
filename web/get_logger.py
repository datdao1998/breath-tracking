import logging
from logstash_async.handler import AsynchronousLogstashHandler




# Get you a test logger
def get_log():
    host = 'localhost'
    port = 5000
    test_logger = logging.getLogger('breath-logger')
    # Set it to whatever level you want - default will be info
    test_logger.setLevel(logging.DEBUG)
    # Create a handler for it
    async_handler = AsynchronousLogstashHandler(host, port, database_path=None)
    # Add the handler to the logger
    test_logger.addHandler(async_handler)

    return test_logger

    # extra = {
    #     "per_id" : "per003",
    #     "duration" : 50,
    #     "normal" : 20,
    #     "deep" : 10,
    #     "other" : 5,
    #     "strong" : 15
    # }
    #
    # test_logger.info('Report',extra=extra)
