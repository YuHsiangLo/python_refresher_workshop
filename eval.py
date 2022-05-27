def create_spans(ipa_syl):
    syll = []

    offset = 0  # offset to correct indices due to '.'
    start = 0  # start index
    ipas = ''  # string representation of syllable
    for i, symbol in enumerate(ipa_syl):
        if symbol == '.':
            syll.append((ipas, start, i + offset))
            start = i + offset
            offset -= 1
            ipas = ''
        elif i == len(ipa_syl) - 1:  # reach the end
            ipas += symbol
            syll.append((ipas, start, i + offset + 1))
        else:
            ipas += symbol

    return syll


def syllabify_baseline(ipa_list):
    syllabified = []
    for i, symbol in enumerate(ipa_list):
        if i % 2 == 1 and i != len(ipa_list) - 1:  # add '.' if the index is odd and not the last one
            syllabified.extend([symbol, '.'])
        else:
            syllabified.append(symbol)

    return syllabified


words = []
with open('data/data_for_eval.tsv', mode='r') as f:
    for line in f:
        orth, ipa, ipa_syl = line.strip().split(sep='\t')
        ipa = ipa.split(sep=' ')
        ipa_syl = ipa_syl.split(sep=' ')

        gold_syll = create_spans(ipa_syl)

        words.append({'orth': orth, 'ipa': ipa, 'syll': gold_syll})

# add baseline syllabification
for word_dict in words:
    word_dict['baseline_syll'] = create_spans(syllabify_baseline(word_dict['ipa']))

# Calculate f1
tp = 0  # true positive
fp = 0  # false positive
fn = 0  # false negative

for word_dict in words:
    gold = word_dict['syll']
    baseline = word_dict['baseline_syll']
    tp += len(set(gold) & set(baseline))
    fp += len(set(baseline) - set(gold))
    fn += len(set(gold) - set(baseline))

pre = tp / (tp + fp)  # precision
rec = tp / (tp + fn)  # recall
f1 = 2 * pre * rec / (pre + rec)
