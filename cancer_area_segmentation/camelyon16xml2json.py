import sys
import os
import argparse
import logging
import glob

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from annotation import Formatter  # noqa

parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to internal json format')
parser.add_argument('--xml_path', default=r'./data/all_xml_tumor_1020', metavar='XML_PATH', type=str,
                    help='Path to the input Camelyon16 xml annotation')
parser.add_argument('--json_path', default=r'/home/sdd/zxy/TCGA_data/json/all_tumor_json_10_20', metavar='JSON_PATH', type=str,
                    help='Path to the output annotation in json format')


def run(args):
    paths = glob.glob(os.path.join(args.xml_path, '*.xml'))
    #ff = os.walk(args.xml_path)
    #paths = []
    #for root, dirs, files in ff:
    #    for file in files:
    #        if os.path.splitext(file)[1] == '.xml':
    #            paths.append(os.path.join(root, file))
    for path in paths:
        print(path)
        json_name = os.path.basename(path)[:-4]
        json_path = os.path.join(args.json_path, json_name + '.json')
        #if os.path.exists(json_path):
        #    continue
        Formatter.camelyon16xml2json(path, json_path)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
