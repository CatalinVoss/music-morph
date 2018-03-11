import numpy as np
import pretty_midi
import mido
from time import sleep
from threading import Lock, Thread
from Queue import Queue # This is thread-safe!

class NeuralDJ():
    def __init__(self, num_notes, bar_quant):
        self.output = mido.open_output('New Port', virtual=True, client_name="Neural-DJ") # mido.open_output('TiMidity:TiMidity port 0 128:0')
        self.bar_buffer = Queue() # thread-safe!
        self.NUM_NOTES = num_notes
        self.BAR_QUANT = bar_quant

    def _play_roll(self, roll):
        """
        Assuming cols are meant to be spaced apart 1/output_tempo seconds
        """
        print('Pingggg')
        for x in range(10):
            self.output.send(mido.Message('note_on', note=72))
            sleep(1)

    def _play_background(self):
        """
        Call on background thread
        """
        while True:
            bar = self.bar_buffer.get()
            roll = None#np.reshape(bar,(self.NUM_NOTES, self.BAR_QUANT))
            self._play_roll(roll)

            self.bar_buffer.task_done()

    def start_playback(self):
        """
        Call on main thread
        """
        self.playback_thread = Thread(target=self._play_background).start()

    def add_bar(self, bar):
        """
        Add a bar that will be played once the current playback is over
        """
        self.bar_buffer.put(bar)

    def finish_playback(self):
        """
        Joins the playback thread and blocks until all bars have been played
        """
        self.bar_buffer.join()