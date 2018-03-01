# music_morph


### Using magenta
```
$ python convert_dir_to_note_sequences.py --input_dir=data/small_test --output_file='data/small_test.tfrecord' --numthreads=4
```


### Alterative frameworks
```
# python-midi

import midi
pattern = midi.read_midifile("/Users/catalin/Downloads/lmd_full/3/3c8a1e5c4f9149b82667f5f8b0b5f8bf.mid")
print(pattern)
```

```
# mido

from mido import MidiFile

mid = MidiFile('/Users/catalin/Downloads/lmd_full/3/3c8a1e5c4f9149b82667f5f8b0b5f8bf.mid')

for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)
```