# move.rAIght

move.rAIght is an AI-powered tool designed to help you analyze and improve your workout form with real-time feedback. Using advanced pose estimation technology, this gym assistant evaluates your form during exercises such as squats and bench presses, ensuring that you get the most out of your workout safely and effectively.

![squat](https://github.com/hbrt-rdzk/move.rAIght/assets/123837698/b30321df-d357-424a-9f5f-6343b4a85e0e)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

A step-by-step series of examples that tell you how to get a development environment running:

Clone the repository to your local machine:

```bash
git clone https://github.com/hbrt-rdzk/move.rAIght.git
```

Navigate to the cloned directory:

```bash
cd move.rAIght
```

Install the necessary dependencies:

```bash
pip install -e .
```

### Usage

To run the Gym Assistant, use the following command:

```bash
python3 scripts/cli.py
```

You have to provide the following arguments:

- `-a`, `--app`: One of the app types (LIVE, VIDEO)
- `-i`, `--input`: Specify the camera number or path to the video file you wish to analyze.
- `--exercise`: Type of analyzing exercise

Optional arguments:
- `-o`, `--output`: Directory path for output data (CSV)
- `--save_results`: If data should be saved or not
- `--loop`: If video should be looped or not


Example of running with a video file:

```bash
python3 scripts/cli.py --app="LIVE" -i="path/to/video.mp4" --exercise="squat" --save_results
```

Example of running with a webcam:

```bash
python3 scripts/cli.py --app="LIVE" -i=0 --exercise="squat" -o="squat.csv" --save_results
```

To analyze the video:
```bash
python3 scripts/cli.py --app="VIDEO" -i="path/to/video.mp4" --exercise="squat" --save_results
```

## Configuration

The `configs/config.yaml` file contains all the configuration parameters that Gym Assistant needs to know before analyzing your workout. Make sure to review and modify it if needed according to your specific requirements.

## Data

The `data/` directory should contain your workout videos categorized by exercise type. This can be used to train the model or for personal record-keeping.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
