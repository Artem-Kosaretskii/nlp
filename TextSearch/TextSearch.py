import os
from rapidfuzz import distance
from rapidfuzz.process import extractOne, cdist
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio
from opensearchpy import OpenSearch
import numpy as np
from dotenv import load_dotenv

np.set_printoptions(legacy='1.25')


def levenshtein(word1: str, word2: str) -> float:
    m = len(word1)
    n = len(word2)
    i, j = 0, 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i

    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j],
                               dp[i][j - 1]) + 1

    return 1 - dp[i][j] / max(m, n)


def opensearch_fn():

    """
    curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{"persistent": {"cluster.blocks.create_index": false}}'
    curl -XPUT "http://localhost:9200/_cluster/settings" -H 'Content-Type: application/json' -d' {"persistent": {"cluster": {"routing": {"allocation.disk.threshold_enabled": false}}}}'
    """
    auth = ('admin', os.getenv('OPENSEARCH_INITIAL_ADMIN_PASSWORD'))
    client = OpenSearch(hosts=[{'host': 'localhost', 'port': 9200}])
    info = client.info()
    print(f"Welcome to {info['version']['distribution']} {info['version']['number']}!")
    index_name = 'my_books'
    index_body = {
        'mappings': {
            'properties': {
                'title': {'type': 'text', 'analyzer': 'english'},
                'author': {'type': 'text', "analyzer": 'english'},
                'year': {'type': 'integer'},
                'plot': {'type': 'text', 'analyzer': 'english'}
            }
        }
    }
    response = client.indices.create(index=index_name, body=index_body)
    print(f'Index created: {response}')
    doc1 = {
        'title': 'Great Expectations',
        'author': 'Charles Dickens',
        'year': 1861,
        'plot': 'Great Expectations is the thirteenth novel by the English author Charles Dickens and his penultimate completed novel. The novel is a bildungsroman and depicts the education of an orphan nicknamed Pip.'
    }
    doc2 = {
        'title': 'A Study in Scarlet',
        'author': 'Arthur Conan Doyle',
        'year': 1887,
        'plot': 'Published in 1887, the story in that novel marks the first appearance of Sherlock Holmes and Dr. Watson, who would go on to become one of the most well-known detective duos in literature.'
    }
    doc3 = {
        'title': 'unknown',
        'author': 'unknown',
        'year': -9999,
        'plot': 'tbd'
    }
    for i, doc in enumerate([doc1, doc2, doc3]):
        print(client.index(index=index_name, id=i, body=doc, refresh=True))
    query = {
        'query': {
            'bool': {
                'must': [
                    {'match': {'plot': 'novel'}},
                ],
                'filter': [
                    {'range': {'year': {'gte': 1800}}}
                ]
            }
        }
    }
    response = client.search(
        index=index_name,
        body=query
    )
    print(response)
    for hit in response['hits']['hits']:
        for k, v in hit.items():
            print(f"{k}: {v}")
    query = {
        'query': {
            'bool': {
                'should': [{
                    'match': {
                        'author': 'Dickens'}
                }, {
                    'range': {
                        'year': {
                            'lte': 1800
                        }
                    }
                }]
            }
        }
    }
    response = client.search(index=index_name, body=query)
    for hit in response['hits']['hits']:
        print(hit["_source"])
        print()

    index_name = 'songs'
    index_body = {
        'mappings': {
            'properties': {
                'title': {'type': 'text', 'analyzer': 'english'},
                'artist': {'type': 'text', "analyzer":'english'},
            }
        }
    }
    response = client.indices.create(index=index_name, body=index_body)
    doc1 = {
        'title': 'Sleeping Sun',
        'artist': 'Nightwish'
    }
    doc2 = {
        'title': 'For Whom The Bell Tolls',
        'artist': 'Metallica'
    }
    doc3 = {
        'title': 'Another Day',
        'artist': 'Dream Theatre'
    }
    for i, doc in enumerate([doc1, doc2, doc3]):
        client.index(index=index_name, id=i, body=doc, refresh=True)
    q = "nightwish"
    query = {
        'query': {
            'match': {
                "artist": {
                    'query': q,
                    "fuzziness": "auto"
                }
            }
        }
    }
    response = client.search(index=index_name, body=query)
    for hit in response['hits']['hits']:
        print(hit["_source"])


def levenshtein_fn():
    print(levenshtein("wave", "waves"))
    print(levenshtein("coca-cola", "cocacola"))
    pair = ("cat", "skate")
    dist = distance.Levenshtein.normalized_distance(pair[0], pair[1])
    print(f"Levenshtein distance for {pair}: {dist}")
    q = "asprin"
    choices = ["erythromycin", "aspirin", "loratadine", "iodine", "avastin", "arthrocin"]
    print(extractOne(q, choices, scorer=Levenshtein.normalized_similarity))
    print(cdist(["asprin", "ioine", "lortadne"], choices))

    brands_list = [
        "Apple", "Samsung Electronics", "Microsoft Corporation",
        "Sony Interactive Entertainment", "Adidas Group", "Nike Inc.",
        "Coca-Cola", "PepsiCo", "Amazon Web Services",
        "Google Cloud ", "Tesla Motors", "Toyota",
        "Mercedes-Benz ", "BMW Group", "Intel Corporation", "IBM ",
        "Oracle Database Systems", "Twitter", "Netflix ",
        "Adobe", "Spotify Music", "Uber ", "Airbnb",
        "LinkedIn"
    ]
    texts = [
        "Yesterday I bought a new phone from Samnsung Electrnics and a laptop from Appel. The quality is excellent!",
        "Microsoft presented new products at the conference, but Sony did not.",
        "Adidas sportswear and Nike shoes are the perfect combination for training.",
        "There were Coca-Cola drinks at the party",
        "Our startup uses Amazon Web Services and GoogleCloud for data storage.",
        "The test drive of the new Tesla and Toyota electric vehicles was successful.",
        "Premium cars from Mercedez-Benz AG and the BMV Group have always been a symbol of status and reliability."
    ]
    threshold = 0.25
    scores = cdist(texts, brands_list) / 100
    # print(np.array(brands_list)[scores[i] > threshold])
    for i, t in enumerate(texts):
        brands = []
        current_text = scores[i]
        for j in range(current_text.shape[0]):
            if current_text[j] > threshold:
                brands.append(brands_list[j])
        print(f'TEXT {i}: {brands}')


def main(levenshtein_run=False, opensearch_run=True):
    load_dotenv()
    if levenshtein_run:
        levenshtein_fn()
    if opensearch_run:
        opensearch_fn()
    return 0


if __name__ == '__main__':
    main(levenshtein_run=False, opensearch_run=True)
