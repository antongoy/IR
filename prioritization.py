import re
import argparse

import numpy as np
import pylab as pl

from random import sample, shuffle
from itertools import combinations
from scipy.cluster.hierarchy import fclusterdata

__author__ = 'anton-goy'


def parse_it():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('general_filename', help="File containing general URLs")
    my_parser.add_argument('examined_filename', help="File containing examined URLs")

    my_args = vars(my_parser.parse_args())

    return my_args['general_filename'], my_args['examined_filename']


def insert_regexp(parse_url, pattern, regexp_object):
    for i, segment in enumerate(parse_url):
        if regexp_object.match(segment):
            parse_url[i] = pattern


def generate_segment_features(url, segment_features):

    for i, segment in enumerate(url):

        if (segment, i) in segment_features:
            segment_features[(segment, i)] += 1
        else:
            segment_features[(segment, i)] = 1


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

    return sorted(url_query)


def generate_query_features(query, query_features):
    for param in query:
        if param in query_features:
            query_features[param] += 1
        else:
            query_features[param] = 1


def parse_urls(urls):
    numerical_pattern = r'[0-9]+'
    numerical_regexp = re.compile(numerical_pattern)

    with_percent_pattern = r'(%[A-Za-z0-9]{2}\+?)+'
    with_percent_regexp = re.compile(with_percent_pattern)

    extension_pattern = '\.[^/?]+'
    extension_regexp = re.compile(extension_pattern)

    segment_features = {}
    query_features = {}
    all_urls = []

    for url in urls:
        if not extension_regexp.search(url) and url[-1] != '/':
            url += '/'

        parse_url = url.split('/')

        if not parse_url[-1] == '':
            query = parse_query(parse_url, numerical_regexp, with_percent_regexp)
            parse_url += query

        insert_regexp(parse_url, numerical_pattern, numerical_regexp)
        insert_regexp(parse_url, with_percent_pattern, with_percent_regexp)

        all_urls.append(parse_url)

        generate_segment_features(parse_url, segment_features)
        #generate_query_features(query, query_features)

    return segment_features, query_features, all_urls


def generate_dataset(all_urls, segment_features):
    data_set = []

    for url in all_urls:
        feature_vector = []
        for segment in segment_features:
            if segment[0][1] < len(url) and url[segment[0][1]] == segment[0][0]:
                feature_vector.append(1)
            else:
                feature_vector.append(0)

        data_set.append(feature_vector)

    return np.array(data_set, dtype=np.int64)


def jaccard_distance(X, Y):
    intersect = sum([x and y for x,y in zip(X, Y)])
    union = sum([x or y for x,y in zip(X, Y)])

    return 1 - intersect / union


def compute_diameter(cluster):
    if len(cluster) == 1:
        return 0

    return max([jaccard_distance(x, y) for x, y in combinations(cluster, 2)])


def main():
    general_filename, examined_filename = parse_it()

    n_urls = 1000
    host_length = 20
    n_features = 70

    with open(general_filename, 'r') as general_file, open(examined_filename, 'r') as examined_file:
        urls = sample([line[host_length:].rstrip('\n') for line in general_file], n_urls // 2) + \
               sample([line[host_length:].rstrip('\n') for line in examined_file], n_urls // 2)
        shuffle(urls)

        segment_features, query_features, all_parse_urls = parse_urls(urls)
        segment_features = segment_features.items()
        segment_features = sorted(segment_features, key=lambda x: x[1], reverse=True)
        segment_features = segment_features[:n_features]

        data_set = generate_dataset(all_parse_urls, segment_features)

        all_parse_urls = np.array(all_parse_urls)
        urls = np.array(urls)

        clusters = fclusterdata(data_set, t=0.27, metric='jaccard', method='single', criterion='distance')
        cluster_labels = list(set(clusters))

        for cluster_label in cluster_labels:
            cluster = all_parse_urls[clusters == cluster_label]

            print(cluster, end='\n\n')

            length = len(max(cluster, key=lambda s: len(s)))

            for i in range(length):
                segments = list(set([url[i] for url in cluster if len(url) - 1 >= i]))
                print(segments)









if __name__ == '__main__':
    main()