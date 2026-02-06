from demo_runner import DemoRunner
from ncnn_runner import NcnnRunner


def main():
    INPUT_FILE = "example_input.npz"
    OUTPUT_FILE = "watermarked_ncnn.wav"
    impl = NcnnRunner("generator.ncnn.param", "generator.ncnn.bin", "detector.ncnn.param", "detector.ncnn.bin")
    demo = DemoRunner(impl)
    demo.run_generator(INPUT_FILE, OUTPUT_FILE)
    demo.run_detector(OUTPUT_FILE)

if __name__ == '__main__':
    main()
