import re
import argparse

import numpy as np

from random import sample

__author__ = 'anton-goy'


def argument_parse():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('general_filename', help="File containing general URLs")
    my_parser.add_argument('examined_filename', help="File containing examined URLs")
    my_parser.add_argument('regexp_filename', help="File containing regexps")

    my_args = vars(my_parser.parse_args())

    return my_args['general_filename'], my_args['examined_filename'], my_args['regexp_filename']


def get_purity(regexps, general_urls, examined_urls):
    purity = 0

    for r in regexps:
        n1 = 0
        n2 = 0

        r_obj = re.compile(r)
        for url in general_urls:
            if r_obj.match(url):
                n1 += 1

        for url in examined_urls:
            if r_obj.match(url):
                n2 += 1

        purity += max(n1, n2)

    return purity / 2000


def main():
    general_filename, examined_filename, regexp_filename = argument_parse()

    with open(general_filename, 'r') as general_file, \
         open(examined_filename, 'r') as examined_file, \
         open(regexp_filename, 'r') as regexp_file:
        print("Open files...")

        all_general_urls = [line.rstrip('\n') for line in general_file]
        all_examined_urls = [line.rstrip('\n') for line in examined_file]
        regexps = [line.rstrip('\n') for line in regexp_file]

        m = 8
        n = 1000
        final_purity = 0

        bootstrap = [(sample(all_general_urls, n), sample(all_examined_urls, n)) for i in range(m)]

        for general_urls, examined_urls in bootstrap:
            final_purity += get_purity(regexps, general_urls, examined_urls)

        final_purity /= m

        print(final_purity)

if __name__ == '__main__':
    main()