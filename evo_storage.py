import jsonpickle
import pickledb


class EvoStorage:
    def __init__(self, dump_file_path='evo_history.db', **kwargs):
        if 'from_file' in kwargs:
            self.db = pickledb.load(kwargs['from_file'], False)
        else:
            self.db = pickledb.load(dump_file_path, False)

    def save_run(self, key, evo_history):
        encoded_history = jsonpickle.encode(evo_history, keys=True)
        self.db.set(key, encoded_history)
        self.db.dump()

    def run_by_key(self, key):
        decoded_history = jsonpickle.decode(self.db.get(key), keys=True)
        return decoded_history
