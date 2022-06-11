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

# week2
train = read_data('./data/train.tsv')
test = read_data('./data/test.tsv')
dev = read_data('./data/dev.tsv')

def evaluate(gold, system):
    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative

    for g, s in zip(gold, system):
        g = [(tuple(syll[0]), syll[1], syll[2]) for syll in g]
        s = [(tuple(syll[0]), syll[1], syll[2]) for syll in s]

        tp += len(set(g) & set(s))
        fp += len(set(s) - set(g))
        fn += len(set(g) - set(s))

    pre = tp / (tp + fp)  # precision
    rec = tp / (tp + fn)  # recall
    return 2 * pre * rec / (pre + rec)

dev_syll = [d['syll'] for d in dev]

baseline_syll = []
with open('./data/dev.tsv', mode='r') as f:
    for line in f:
        _, ipa, _ = line.strip().split(sep='\t')
        word_dict = baseline(ipa.split(sep=' '))
        baseline_syll.append(word_dict)

evaluate(dev_syll, baseline_syll)

IPAS = set()

with open('data/train.tsv') as f:
    for line in f:
        orth, trans, syll = line.strip().split(sep='\t')
        IPAS.update(trans.split(sep=' '))

with open('data/dev.tsv') as f:
    for line in f:
        orth, trans, syll = line.strip().split(sep='\t')
        IPAS.update(trans.split(sep=' '))

CONS = {'f', 'ʃ', 'j', 'l', 'ɹ', 'h', 'b', 'v', 'w', 'd',
        'ð', 'k', 'x', 's', 'ʒ', 'θ', 't͡ʃ', 't', 'n', 'p',
        'd͡ʒ', 'ŋ', 'm', 'ɡ', 'z'}
VOWEL = {'ʌ', 'm̩', 'n̩', 'ɒ̃ː', 'l̩', 'ɪə', 'ŋ̩', 'ɒ', 'ə', 'aʊ',
         'ɪ', 'iː', 'æ̃ː', 'ɑː', 'æ', 'uː', 'ɛ', 'ɜː', 'ʊ', 'əʊ',
         'ɔː', 'aɪ', 'ɔɪ', 'eɪ', 'ʊə', 'ɑ̃ː', 'ɛə'}


def cv_syll(ipas):
    sylls = []
    current_syll = []

    start = 0
    for index, symbol in enumerate(ipas):
        current_syll.append(symbol)
        if symbol in VOWEL:
            sylls.append((current_syll, start, index + 1))

            start = index + 1
            current_syll = []

        if index == len(ipas) - 1:
            if sylls == []:
                sylls.append(([], start, index + 1))

            sylls[-1] = (sylls[-1][0] + current_syll,
                         sylls[-1][1],
                         index + 1)

    return sylls


dev_syll = [d['syll'] for d in dev]

baseline_syll = []
with open('./data/dev.tsv', mode='r') as f:
    for line in f:
        _, ipa, _ = line.strip().split(sep='\t')
        word_dict = cv_syll(ipa.split(sep=' '))
        baseline_syll.append(word_dict)

evaluate(dev_syll, baseline_syll)

STOP = {'b', 'd', 'k', 't', 'p', 'ɡ'}
FRIC = {'f', 'ʃ', 'h', 'v', 'ð', 'x', 's', 'ʒ', 'θ', 'z'}


def clever_syll(ipas):
    sylls = []
    current_syll = []
    seen_stop = False

    start = 0
    for index, symbol in enumerate(ipas):
        current_syll.append(symbol)
        if symbol in STOP:
            if seen_stop and symbol in STOP | FRIC:
                sylls[-1] = (sylls[-1][0] + [current_syll[0]],
                             sylls[-1][1],
                             index)
                current_syll = current_syll[1:]
                start += 1

            seen_stop = not seen_stop

        if symbol in VOWEL:
            sylls.append((current_syll, start, index + 1))

            seen_stop = False
            start = index + 1
            current_syll = []

        if index == len(ipas) - 1:
            if sylls == []:
                sylls.append(([], start, index + 1))

            sylls[-1] = (sylls[-1][0] + current_syll,
                         sylls[-1][1],
                         index + 1)

    return sylls


dev_syll = [d['syll'] for d in dev]

baseline_syll = []
with open('./data/dev.tsv', mode='r') as f:
    for line in f:
        _, ipa, _ = line.strip().split(sep='\t')
        word_dict = clever_syll(ipa.split(sep=' '))
        baseline_syll.append(word_dict)

evaluate(dev_syll, baseline_syll)

print(cv_syll(['ə', 'd', 'æ', 'p', 't', 'ə']))
print(clever_syll(['ə', 'd', 'æ', 'p', 's', 't', 'ə']))
print(clever_syll(['z']))