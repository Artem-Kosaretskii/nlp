from collections import Counter, OrderedDict


def get_pairs(symbols):
    """
    ["h","e","l","l","o"] -> [("h","e"),("e","l"),("l","l"),("l","o")].
    """
    pairs = []
    for i in range(len(symbols) - 1):
        pairs.append((symbols[i], symbols[i + 1]))
    return pairs


def best_pair(words):
    """
    [["a","b","a","b","c"], ["a","b","c"]] -> (best pair, freq) or (None, 0).
    """
    count = {}
    for w in words:
        pairs = get_pairs(w)
        for p in pairs:
            if count.get(p, None):
                count[p] += 1
            else:
                count[p] = 1
    count = OrderedDict(sorted(count.items(), key=lambda item: item[1], reverse=True))
    return next(iter(count)), count[next(iter(count))] if len(count) != 0 else (None, 0)


def best_pair_with_counter(words):
    pair_counts = Counter()

    for symbols in words:
        pair_counts.update(get_pairs(symbols))

    if not pair_counts:
        return None, 0

    pair, freq = pair_counts.most_common(1)[0]
    return pair, freq


def merge_pair(symbols, pair):
    """
    ["h","e","l","l","o"], pair=("l","l")-> ["h","e","ll","o"].
    """
    i, new_symbols = 0, []
    while i < len(symbols):
        if i < len(symbols)-1:
            if symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                new_symbols.append(pair[0]+pair[1])
                i += 2
                continue
        new_symbols.append(symbols[i])
        i += 1

    return new_symbols

def main():

    words = [
        ["a", "b", "c", "a"],
        ["a", "b", "c"],
        ["b", "c", "a"]
    ]

    pair, freq = best_pair(words)
    print(pair, freq)

    words_merged = [merge_pair(w, pair) for w in words]
    print(words_merged)


if __name__ == '__main__':
    main()
