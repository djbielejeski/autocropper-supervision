import argparse
import os

class AutoCropperArguments:
    def __init__(
            self,
            images_directory,
            results_folder="results",
            crop_ratio=(3, 4),
            person_percent_detection_cutoff=0.075,
            person_padding_percent=0.02,
    ):
        self.results_folder = results_folder
        self.images_directory = images_directory
        self.crop_ratio = crop_ratio
        self.person_percent_detection_cutoff = person_percent_detection_cutoff
        self.person_padding_percent = person_padding_percent

        os.makedirs(self.results_folder, exist_ok=True)


def parse_arguments() -> AutoCropperArguments:
    def _get_parser(**parser_kwargs):
        def tuple_type(strings):
            strings = strings.replace("(", "").replace(")", "")
            mapped_int = map(int, strings.split(","))
            return tuple(mapped_int)

        parser = argparse.ArgumentParser(**parser_kwargs)

        parser.add_argument(
            "--images_directory",
            type=str,
            required=True,
            help="Directory containing the images you want to process.  Should be a directory path to a folder containing *.png or *.jpg files."
        )

        parser.add_argument(
            "--results_folder",
            type=str,
            required=False,
            default="results",
            help="Path to where you want to save the output files"
        )

        parser.add_argument(
            "--crop_ratio",
            type=tuple_type,
            default="(3,4)",
            required=False,
            help="\"(Width,Height)\" ratio for cropping the output files.  Supports \"(1,1)\", \"(2,3)\", and \"(3,4)\"."
        )

        parser.add_argument(
            "--person_percent_detection_cutoff",
            type=float,
            required=False,
            default=0.075,
            help="Percentage of the image that needs to contain a person in order for it to be detected."
        )

        parser.add_argument(
            "--person_padding_percent",
            type=float,
            required=False,
            default=0.02,
            help="Percentage of the detected person in pixels that we adding padding around."
        )

        return parser

    parser = _get_parser()
    opt, unknown = parser.parse_known_args()

    config = AutoCropperArguments(
        images_directory=opt.images_directory,
        results_folder=opt.results_folder,
        crop_ratio=opt.crop_ratio,
        person_percent_detection_cutoff=opt.person_percent_detection_cutoff,
        person_padding_percent=opt.person_padding_percent,
    )

    return config
