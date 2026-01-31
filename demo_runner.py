import sys
import numpy as np
import soundfile
from tqdm.auto import tqdm

from inference_runner import InferenceRunner

class DemoRunner:
    MODEL_INPUT_SECONDS = 2.0
    WATERMARK_STRENGTH = 1.0

    def __init__(self, input_file: str, output_file: str, impl: InferenceRunner):
        self.input_file = input_file
        self.output_file = output_file
        self.impl = impl

    def run_generator(self):
        tqdm.write(f"Running watermark generator from {self.input_file} to {self.output_file}...")
        audio, fs = soundfile.read(self.input_file, dtype='float32')
        frame_length = int(self.MODEL_INPUT_SECONDS * fs)
        audio = audio[0:(len(audio) - len(audio) % frame_length)]
        input_length = len(audio)
        ret = np.ndarray(shape=(int(input_length / 3)), dtype=np.float32)
        audio = audio[None, None, None, :]
        input_index = 0
        output_index = 0

        for _ in tqdm(range(input_length // frame_length), file=sys.stdout, leave=True):
            input_seg = audio[:, :, :, input_index:input_index + frame_length]
            input_index = input_index + frame_length
            output = self.impl.run_generator(input_seg, self.WATERMARK_STRENGTH)
            ret[output_index: output_index + len(output)] = output
            output_index = output_index + len(output)
        soundfile.write(self.output_file, ret, fs // 3)
        tqdm.write(f"Watermark added and wrote to {self.output_file}.")

    def run_detector(self):
        tqdm.write(f"Running watermark detector from {self.output_file}...")
        audio, fs = soundfile.read(self.output_file, dtype='float32')
        frame_length = int(self.MODEL_INPUT_SECONDS * fs)
        audio = audio[0:(len(audio) - len(audio) % frame_length)]
        input_length = len(audio)
        audio = audio[None, None, None, :]
        input_index = 0
        ret = np.ndarray(shape=(int(input_length // frame_length)), dtype=np.float32)
        for i in tqdm(range(input_length // frame_length), file=sys.stdout, leave=True):
            input_seg = audio[:, :, :, input_index:input_index + frame_length]
            input_index = input_index + frame_length
            output = self.impl.run_detector(input_seg)
            ret[i] = output
        print(ret.flatten())
