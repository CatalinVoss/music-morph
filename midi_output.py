import numpy as np
import pretty_midi
import mido
from time import sleep
from threading import Lock, Thread
from Queue import Queue # This is thread-safe!


NOTES_OFF_AT_BAR_END = False

class NeuralDJ():
    def __init__(self, num_notes, bar_quant, roll_window):
        self.output = mido.open_output('DJ Port', virtual=True, client_name="Neural-DJ") # mido.open_output('TiMidity:TiMidity port 0 128:0')
        self.bar_buffer = Queue() # thread-safe!
        self.NUM_NOTES = num_notes
        self.BAR_QUANT = bar_quant
        self.ROLL_WINDOW = roll_window

    def _play_roll(self, roll, time_tick=0.04):
        """
        Assuming cols are meant to be spaced apart time_tick seconds
        """
        note_status = np.zeros(self.NUM_NOTES)

        # Iterate over time steps -- columns
        for col in roll.T:
            for idx, velocity in enumerate(col):
                pitch = idx + self.ROLL_WINDOW[0]
                if velocity > 0 and note_status[idx] == 0:
                    # Note on
                    self.output.send(mido.Message('note_on', note=pitch, velocity=velocity))
                if velocity == 0 and note_status[idx] > 0:
                    # Note off (note_on with velocity 0)
                    self.output.send(mido.Message('note_on', note=pitch, velocity=0))
                note_status[idx] = velocity
            sleep(time_tick)

        if NOTES_OFF_AT_BAR_END:
            # All notes off at the end of the bar (next one can turn them back on...)
            for idx in range(roll.shape[0]):
                pitch = idx + self.ROLL_WINDOW[0]
                self.output.send(mido.Message('note_on', note=pitch, velocity=0))


    def _play_background(self):
        """
        Call on background thread
        """
        while True:
            bar = self.bar_buffer.get()
            roll = np.reshape(bar,(self.NUM_NOTES, self.BAR_QUANT))
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