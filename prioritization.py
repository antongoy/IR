import re
import argparse

import numpy as np

from random import sample, shuffle
from itertools import combinations
from scipy.cluster.hierarchy import fclusterdata

__author__ = 'anton-goy'


def argument_parsing():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('general_filename', help="File containing general URLs")
    my_parser.add_argument('examined_filename', help="File containing examined URLs")

    my_args = vars(my_parser.parse_args())

    return my_args['general_filename'], my_args['examined_filename']


def insert_regexp(parse_url, pattern, regexp_object):
    for i, segment in enumerate(parse_url):
        if regexp_object.match(segment):
            parse_url[i] = pattern


def generate_features(parsed_urls):
    features = {}

    for url in parsed_urls:
        for i, segment in enumerate(url):

            if (segment, i) in features:
                features[(segment, i)] += 1
            else:
                features[(segment, i)] = 1
    return features


def parse_query(parse_url, num_regexp, percent_regexp):
    url_query = parse_url[-1].split('?')

    if len(url_query) == 1:
        return ['']

    parse_url[-1] = url_query[0]
    url_query = url_query[1]
    url_query = url_query.split('&')

    for i, param in enumerate(url_query):
        match_object = num_regexp.search(param)
        if match_object:
            url_query[i] = url_query[i][:match_object.start()] + '[0-9]+'

        match_object = percent_regexp.search(param)
        if match_object:
            url_query[i] = url_query[i][:match_object.start()] + '(%[A-Za-z0-9]{2}\+?)+'

    return sorted(url_query, reverse=True)


def generate_query_features(query, query_features):
    for param in query:
        if param in query_features:
            query_features[param] += 1
        else:
            query_features[param] = 1


def parse_urls(urls):
    numerical_pattern = r'[0-9]+'
    numerical_regexp = re.compile(numerical_pattern)

    with_percent_pattern = r'[^/]+%[^/]+'
    with_percent_regexp = re.compile(with_percent_pattern)

    extension_pattern = '\.[^/?]+'
    extension_regexp = re.compile(extension_pattern)

    flv_extension = '.flv'

    all_parsed_urls = []

    for url in urls:
        url_dict = {}
        if not extension_regexp.search(url) and url[-1] != '/':
            url += '/'

        parse_url = url.split('/')

        if parse_url[-1].endswith(flv_extension):
            parse_url[-1] = r'[^/]+.flv'

        if not parse_url[-1] == '':
            query = parse_query(parse_url, numerical_regexp, with_percent_regexp)
            parse_url += query

        insert_regexp(parse_url, numerical_pattern, numerical_regexp)
        insert_regexp(parse_url, with_percent_pattern, with_percent_regexp)

        all_parsed_urls.append(parse_url)

    return all_parsed_urls


def generate_dataset(all_urls, features):
    data_set = []

    for url in all_urls:
        feature_vector = []
        for segment in features:
            if segment[0][1] < len(url) and url[segment[0][1]] == segment[0][0]:
                feature_vector.append(1)
            else:
                feature_vector.append(0)

        data_set.append(feature_vector)

    return np.array(data_set, dtype=np.int64)


def jaccard_distance(X, Y):
    intersect = sum([x and y for x, y in zip(X, Y)])
    union = sum([x or y for x, y in zip(X, Y)])

    return 1 - intersect / union


def compute_diameter(cluster):
    if len(cluster) == 1:
        return 0

    return max([jaccard_distance(x, y) for x, y in combinations(cluster, 2)])


def main():
    general_filename, examined_filename = argument_parsing()

    n_urls = 2000
    host_length = 20
    n_features = 100

    with open(general_filename, 'r') as general_file, open(examined_filename, 'r') as examined_file:
        print("Open files...")

        general_urls = [line[host_length:].rstrip('\n') for line in general_file]
        examined_urls = [line[host_length:].rstrip('\n') for line in examined_file]

        print("Start of parsing general urls...")
        general_parsed_urls = parse_urls(general_urls)

        print("Start of parsing examined urls")
        examined_parsed_urls = parse_urls(examined_urls)

        clustered_parsed_urls = sample(general_parsed_urls, n_urls // 2) + \
                                sample(examined_parsed_urls, n_urls // 2)
        shuffle(clustered_parsed_urls)

        print("Generate features...")
        all_features = generate_features(clustered_parsed_urls)

        print("Sort features...")
        all_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:n_features]

        all_parsed_urls = sample(general_parsed_urls + examined_parsed_urls, 4000)
        shuffle(all_parsed_urls)

        print("Generate dataset...")
        data_set = generate_dataset(all_parsed_urls, all_features)

        all_parsed_urls = np.array(all_parsed_urls)

        print("Start clustering...")
        clusters = fclusterdata(data_set, t=0.2, metric='jaccard', method='single', criterion='distance')
        cluster_labels = list(set(clusters))

        regexps = []

        for cluster_label in cluster_labels:
            cluster = all_parsed_urls[clusters == cluster_label]

            print(all_parsed_urls[clusters == cluster_label], end='\n\n')

            length = len(max(cluster, key=lambda s: len(s)))

            pattern = '/'

            for i in range(length):
                segments = list(set([url[i] for url in cluster if len(url) - 1 >= i]))
                if len(segments) > 1:
                    pattern += r'[^/]+/'
                else:
                    pattern += segments[0] + '/'
            regexps.append(pattern)

        #for r in regexps:
        #    print(r)


if __name__ == '__main__':
    main()