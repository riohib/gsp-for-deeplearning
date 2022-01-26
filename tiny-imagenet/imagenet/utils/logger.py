import logging
import os

class Logger():
    def __init__(self, log_dir, logfile_name):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f" \n Created Logdir at:{log_dir} \n")
    
        self.logger = logging.getLogger(logfile_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger_path = log_dir + '/' + logfile_name + '.log'
        self.formatter = logging.Formatter('%(levelname)s:%(name)s:     %(message)s')
    
        self.get_file_handler(self.logger_path, self.formatter)
    
    def get_file_handler(self, logger_path, formatter):
        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        return file_handler
    
    def get_stream_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(stream_handler)
        return stream_handler
    
    def get_file_logger(self):
        print("\n RETURNED FILE LOGGER! \n")
        return self.logger
        
    
    def get_stream_logger(self):
        self.get_stream_handler()
        return self.logger
    
    def log_cmd_arguments(self, args):
        self.log_separation(2)
        self.logger.info(f"All the arguments used are: ")
        for arg in vars(args):
            self.logger.info(f"{arg : <20}: {getattr(args, arg)}")
        self.log_separation(2)

    def log_separation(self, repeat_times=1):
        for i in range(repeat_times):
            self.logger.info(80*"-")