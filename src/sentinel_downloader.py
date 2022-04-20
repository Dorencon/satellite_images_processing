import os
import sys
import argparse
from sentinelhub import (
    SHConfig,
    MimeType,
    CRS,
    BBox,
    SentinelHubRequest,
    DataCollection,
    bbox_to_dimensions,
)


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Sentinel-2 downloader', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-i', '--id', type=str, required=False,
                      help='Your sh_client_id from OAuth client profile on page https://apps.sentinel-hub.com/dashboard/#/account/settings.')
    args.add_argument('-s', '--secret', type=str, required=False,
                      help='Your sh_client_secret from OAuth client profile on page https://apps.sentinel-hub.com/dashboard/#/account/settings.')
    args.add_argument('-b', '--bbox', type=str, required=True,
                      help='Area to observe in format "x1 y1 x2 y2" where "x1" and "y1" - lower left longitude and lattitude, "x2" and "y2" - upper right longitude and lattitude')
    args.add_argument('-sd', '--start', type=str, required=True,
                      help='Start date in format YYYY-MM-DD.')
    args.add_argument('-e', '--end', type=str, required=True,
                      help='End date in format YYYY-MM-DD.')
    args.add_argument('-r', '--resolution', type=float, required=True,
                      help='Pixel resolution (in meters).')
    return parser


def main():

    args = build_argparser().parse_args()

    config = SHConfig()

    if not args.id or not args.secret:
        if not config.sh_client_id or not config.sh_client_secret:
            print('Please configure your sh_client_id and sh_client_secret to enable downloading requests')
            print('You can do this by editing config.json file of sentinelhub or by passing --id and --secret arguments just once')
            return
    else:
        config.sh_client_id = args.id
        config.sh_client_secret = args.secret
        config.save()

    bbox = args.bbox
    bbox = bbox.split()
    bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=args.resolution)

    print(f"Image shape at {args.resolution} m resolution: {size} pixels")

    evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
                }],
                output: {
                    bands: 13,
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B01,
                    sample.B02,
                    sample.B03,
                    sample.B04,
                    sample.B05,
                    sample.B06,
                    sample.B07,
                    sample.B08,
                    sample.B8A,
                    sample.B09,
                    sample.B10,
                    sample.B11,
                    sample.B12];
        }
    """

    request = SentinelHubRequest(
        data_folder=os.path.join(os.path.abspath(""), 'sentinel_downloaded'),
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                # TODO check others collections
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(args.start, args.end),
                mosaicking_order="leastCC",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],

        bbox=bbox,
        size=size,
        config=config,
    )
    request.save_data()

    # Debug
    # import numpy as np
    # import matplotlib.pyplot as plt
    # img = request.get_data()[0]
    # plt.imshow(np.concatenate(
    #     (img[:, :, 3:4]*3.5/255, img[:, :, 2:3]*3.5/255, img[:, :, 1:2]*3.5/255), axis=2))
    # plt.show()


if __name__ == '__main__':
    sys.exit(main() or 0)
