# move.rAIght

move.rAIght is an AI-powered tool designed to help you analyze and improve your workout form with real-time feedback. Using advanced pose estimation technology, this gym assistant evaluates your form during exercises such as squats and bench presses, ensuring that you get the most out of your workout safely and effectively.
![video](https://github.com/hbrt-rdzk/move.rAIght/assets/123837698/3509ca5e-7bd6-4e7c-b4d1-5589febbb60c)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

```bash
pip install -r requirements.txt
```
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
pip install -r requirements.txt
```

### Usage

To run the Gym Assistant, use the following command:

```bash
python3 src/app/app.py
```

You can provide the following arguments:

- `-i`, `--input`: Specify the camera number or path to the video file you wish to analyze.
- `--loop`: Enable this flag to loop the video analysis.

Example of running with a video file:

```bash
python3 src/app/app.py -i path/to/video.mp4
```

Example of running with a webcam:

```bash
python3 src/app/app.py -i 0
```

To loop the video:

```bash
python3 src/app/app.py -i path/to/video.mp4 --loop
```

## Configuration

The `configs/config.yaml` file contains all the configuration parameters that Gym Assistant needs to know before analyzing your workout. Make sure to review and modify it if needed according to your specific requirements.

## Data

The `data/` directory should contain your workout videos categorized by exercise type. This can be used to train the model or for personal record-keeping.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://yourrepositoryurl.com/tags).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
