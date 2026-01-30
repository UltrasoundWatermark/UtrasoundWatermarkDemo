from demo_runner import DemoRunner
from ncnn_runner import NcnnRunner


def main():
    INPUT_FILE = "example_input.wav"
    OUTPUT_FILE = "watermarked_ncnn.wav"
    impl = NcnnRunner("generator.ncnn.param", "generator.ncnn.bin", "detector.ncnn.param", "detector.ncnn.bin")
    demo = DemoRunner(INPUT_FILE, OUTPUT_FILE, impl)
    demo.run_generator()
    demo.run_detector()


if __name__ == '__main__':
    main()
