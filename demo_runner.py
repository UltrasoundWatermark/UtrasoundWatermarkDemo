import sys
import numpy as np
import soundfile
from tqdm.auto import tqdm

from inference_runner import InferenceRunner


class DemoRunner:
    MODEL_INPUT_SECONDS = 2.0
    WATERMARK_STRENGTH = 0.2

    def __init__(self, impl: InferenceRunner):
        self.impl = impl

    def run_generator(self, input_file: str, output_file: str):
        tqdm.write(f"Running watermark generator from {input_file} to {output_file}...")
        if input_file.endswith(".wav"):
            audio, fs = soundfile.read(input_file, dtype='float32')
        elif input_file.endswith(".npz"):
            with np.load(input_file, allow_pickle=False) as data:
                audio, fs = data["audio"], data["fs"]
        frame_length = int(self.MODEL_INPUT_SECONDS * fs)
        audio = audio[0:(len(audio) - len(audio) % frame_length)]
        input_length = len(audio)
        ret = np.ndarray(shape=(int(input_length / 3)), dtype=np.float32)
        input_index = 0
        output_index = 0

        for _ in tqdm(range(input_length // frame_length), file=sys.stdout, leave=True):
            input_seg = audio[input_index:input_index + frame_length]
            input_index = input_index + frame_length
            norm_coeff = 1.0 / input_seg.max() * 0.99
            input_seg = input_seg * norm_coeff
            output = self.impl.run_generator(input_seg, self.WATERMARK_STRENGTH)
            output = output / norm_coeff
            ret[output_index: output_index + len(output)] = output
            output_index = output_index + len(output)
        soundfile.write(output_file, ret, fs // 3, subtype='FLOAT')
        tqdm.write(f"Watermark added and wrote to {output_file}.")

    def run_detector(self, input_file: str):
        tqdm.write(f"Running watermark detector from {input_file}...")
        audio, fs = soundfile.read(input_file, dtype='float32')
        frame_length = int(self.MODEL_INPUT_SECONDS * fs)
        audio = audio[0:(len(audio) - len(audio) % frame_length)]
        input_length = len(audio)
        input_index = 0
        ret = np.ndarray(shape=(int(input_length // frame_length)), dtype=np.float32)
        for i in tqdm(range(input_length // frame_length), file=sys.stdout, leave=True):
            input_seg = audio[input_index:input_index + frame_length]
            input_index = input_index + frame_length
            input_seg = input_seg / input_seg.max() * 0.99
            output = self.impl.run_detector(input_seg)
            ret[i] = output
        print(ret.flatten())
        print(f"Avg. Prob. {ret.flatten().mean()}")
