import re
import argparse

import numpy as np

from random import sample, shuffle
from itertools import permutations
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
        if regexp_object.fullmatch(segment):
            parse_url[i] = pattern


def generate_features(parsed_urls):
    features = {}

    for url in parsed_urls:
        for i, segment in enumerate(url['pos_feature']):
            if ('pos_feature', segment, i) in features:
                features[('pos_feature', segment, i)] += 1
            else:
                features[('pos_feature', segment, i)] = 1

        for i, segment in enumerate(url['query_feature']):
            if ('query_feature', segment) in features:
                features[('query_feature', segment)] += 1
            else:
                features[('query_feature', segment)] = 1

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
            url_query[i] = url_query[i][:match_object.start()] + '[^/]*%[^/]+'

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

    with_percent_pattern = r'[^/=]*%[^/]+'
    with_percent_regexp = re.compile(with_percent_pattern)

    extension_pattern = '\.[^/?]+'
    extension_regexp = re.compile(extension_pattern)

    date_pattern = r'\d{4}\-\d{2}\-\d{2}'
    date_regexp = re.compile(date_pattern)

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
            url_dict['query_feature'] = query
        else:
            url_dict['query_feature'] = ['']

        insert_regexp(parse_url, date_pattern, date_regexp)
        insert_regexp(parse_url, numerical_pattern, numerical_regexp)
        insert_regexp(parse_url, with_percent_pattern, with_percent_regexp)

        url_dict['pos_feature'] = parse_url

        all_parsed_urls.append(url_dict)

    return all_parsed_urls


def generate_dataset(all_urls, features):
    data_set = []

    for url in all_urls:
        feature_vector = []
        for feature in features:
            if feature[0][0] == 'pos_feature':
                if feature[0][2] < len(url) and url['pos_feature'][feature[0][2]] == feature[0][1]:
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)
            if feature[0][0] == 'query_feature':
                if feature[0][1] in url['query_feature']:
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)

        data_set.append(feature_vector)

    return np.array(data_set, dtype=np.int64)


def main():
    general_filename, examined_filename = argument_parsing()

    n_urls = 2000
    host_length = 20
    n_features = 150

    with open(general_filename, 'r') as general_file, open(examined_filename, 'r') as examined_file:
        print("Open files...")

        general_urls = [line[host_length:].rstrip('\n') for line in general_file]
        examined_urls = [line[host_length:].rstrip('\n') for line in examined_file]

        print("Start of parsing general urls...")
        general_parsed_urls = parse_urls(general_urls)

        print("Start of parsing examined urls...")
        examined_parsed_urls = parse_urls(examined_urls)

        clustered_parsed_urls = sample(general_parsed_urls, n_urls // 2) + \
                                sample(examined_parsed_urls, n_urls // 2)
        shuffle(clustered_parsed_urls)

        print("Generate features...")
        all_features = generate_features(clustered_parsed_urls)

        print("Sort features...")
        all_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:n_features]

        all_parsed_urls = sample(general_parsed_urls, 1000) + sample(examined_parsed_urls, 1000)
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
            length = len(max(cluster, key=lambda s: len(s['pos_feature']))['pos_feature'])
            pattern = 'http://kinopoisk.ru/'

            for i in range(length):
                segments = list(set([url['pos_feature'][i] if len(url['pos_feature']) - 1 >= i else '' for url in cluster]))
                if len(segments) > 1:
                    if '' in segments and len(segments) > 2:
                        pattern += r'[^/]*/?'
                    if '' in segments and len(segments) == 2:
                        j = segments.index('')
                        pattern += '(' + segments[1 - j] + '/)?'
                    if not '' in segments:
                        pattern += r'[^/]+/'
                else:
                    if not(i == length - 1 and segments[0] == ''):
                        pattern += segments[0] + '/'

            params = list(set([param for url in cluster for param in url['query_feature']]))

            if not '' in params:
                pattern = pattern[:-1]
                pattern += '\?'

                if len(params) > 1:
                    r = '('
                    for s in list(permutations(params, len(params))):
                        t = ''
                        for q in s:
                            t += q + '&'
                        t = t[:-1]
                        r += t + '|'
                    r = r[:-1] + ')'
                    pattern += r
                else:
                    pattern += params[0]

            regexps.append(pattern + '\n')

        with open('regexps.txt', 'w') as output:
            output.writelines(regexps)
            output.flush()


if __name__ == '__main__':
    main()