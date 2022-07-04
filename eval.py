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
            # the end of the current syllable, append it!
            sylls.append((current_syll, start, index + 1))

            start = index + 1
            current_syll = []

        if index == len(ipas) - 1:  # end of ipa list

            # handle cases like [z] or [d]
            if sylls == []:
                sylls.append(([], start, index + 1))

            # ə ɹ ɛ s t
            # sylls = [[ə], [ɹ, ɛ]]
            # current_syll = [[s, t]]
            # take out the last syllable [ɹ, ɛ], combine it with current_syll

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

with open('./data/dev.tsv', mode='r') as f:
    for line in f:
        _, _, syll = line.strip().split(sep='\t')

        delimited = []
        for i, symbol in enumerate(syll[:-1]):
            if symbol != '.' and syll[i+1] != '.':
                delimited.extend([symbol, '.'])
            else:
                delimited.append(symbol)
        delimited.append(syll[-1])

        print(delimited)

from collections import Counter
from pprint import pprint
from math import log

bound_counter_1 = Counter()
no_bound_counter_1 = Counter()

bound_counter_2 = Counter()
no_bound_counter_2 = Counter()

cv_bound_counter_1 = Counter()
cv_no_bound_counter_1 = Counter()

cv_bound_counter_2 = Counter()
cv_no_bound_counter_2 = Counter()

NUM_IPA = len(IPAS)

bound_count = 0
no_bound_count = 0


def determine_cv(ipa):
    if ipa in CONS:
        return 'C'
    elif ipa in VOWEL:
        return 'V'
    else:
        return '#'


with open('./data/train.tsv', mode='r') as f:
    for line in f:
        _, _, syll = line.strip().split(sep='\t')
        syll = syll.split(sep=' ')

        delimited = []
        for i, symbol in enumerate(syll[:-1]):
            if symbol != '.' and syll[i + 1] != '.':
                delimited.extend([symbol, '|'])
            else:
                delimited.append(symbol)

        delimited.append(syll[-1])
        delimited = ['#', '|'] + delimited + ['|', '#']

        for i in range(2, len(delimited) - 2):
            if delimited[i] == '|':
                no_bound_count += 1

                cv_no_bound_counter_2[
                    (
                    determine_cv(delimited[i - 3]), determine_cv(delimited[i - 1]), '_', determine_cv(delimited[i + 1]),
                    determine_cv(delimited[i + 3]))
                ] += 1
                cv_no_bound_counter_1[
                    (determine_cv(delimited[i - 1]), '_', determine_cv(delimited[i + 1]))
                ] += 1

                no_bound_counter_2[
                    (delimited[i - 3], delimited[i - 1], '_', delimited[i + 1], delimited[i + 3])
                ] += 1
                no_bound_counter_1[
                    (delimited[i - 1], '_', delimited[i + 1])
                ] += 1
            elif delimited[i] == '.':
                bound_count += 1

                cv_bound_counter_2[
                    (
                    determine_cv(delimited[i - 3]), determine_cv(delimited[i - 1]), '_', determine_cv(delimited[i + 1]),
                    determine_cv(delimited[i + 3]))
                ] += 1
                cv_bound_counter_1[
                    (determine_cv(delimited[i - 1]), '_', determine_cv(delimited[i + 1]))
                ] += 1

                bound_counter_2[
                    (delimited[i - 3], delimited[i - 1], '_', delimited[i + 1], delimited[i + 3])
                ] += 1
                bound_counter_1[
                    (delimited[i - 1], '_', delimited[i + 1])
                ] += 1

pprint(cv_no_bound_counter_2)

with open('./data/dev.tsv', mode='r') as f:
    for line in f:
        _, _, syll = line.strip().split(sep='\t')
        syll = syll.split(sep=' ')

        delimited = []
        for i, symbol in enumerate(syll[:-1]):
            if symbol != '.' and syll[i + 1] != '.':
                delimited.extend([symbol, '|'])
            else:
                delimited.append(symbol)

        delimited.append(syll[-1])
        delimited = ['#', '|'] + delimited + ['|', '#']

        for i in range(2, len(delimited) - 2):
            if delimited[i] == '|':
                no_bound_count += 1

                cv_no_bound_counter_2[
                    (
                    determine_cv(delimited[i - 3]), determine_cv(delimited[i - 1]), '_', determine_cv(delimited[i + 1]),
                    determine_cv(delimited[i + 3]))
                ] += 1
                cv_no_bound_counter_1[
                    (determine_cv(delimited[i - 1]), '_', determine_cv(delimited[i + 1]))
                ] += 1

                no_bound_counter_2[
                    (delimited[i - 3], delimited[i - 1], '_', delimited[i + 1], delimited[i + 3])
                ] += 1
                no_bound_counter_1[
                    (delimited[i - 1], '_', delimited[i + 1])
                ] += 1
            elif delimited[i] == '.':
                bound_count += 1

                cv_bound_counter_2[
                    (
                    determine_cv(delimited[i - 3]), determine_cv(delimited[i - 1]), '_', determine_cv(delimited[i + 1]),
                    determine_cv(delimited[i + 3]))
                ] += 1
                cv_bound_counter_1[
                    (determine_cv(delimited[i - 1]), '_', determine_cv(delimited[i + 1]))
                ] += 1

                bound_counter_2[
                    (delimited[i - 3], delimited[i - 1], '_', delimited[i + 1], delimited[i + 3])
                ] += 1
                bound_counter_1[
                    (delimited[i - 1], '_', delimited[i + 1])
                ] += 1

# pprint(bound_counter_1)

demon_b_counter_2 = sum(bound_counter_2.values()) + (NUM_IPA + 1) ** 4
demon_b_counter_1 = sum(bound_counter_1.values()) + (NUM_IPA + 1) ** 2

demon_n_counter_2 = sum(no_bound_counter_2.values()) + (NUM_IPA + 1) ** 4
demon_n_counter_1 = sum(no_bound_counter_1.values()) + (NUM_IPA + 1) ** 2

demon_cv_b_counter_2 = sum(cv_bound_counter_2.values()) + 3 ** 4
demon_cv_b_counter_1 = sum(cv_bound_counter_1.values()) + 3 ** 2

demon_cv_n_counter_2 = sum(cv_no_bound_counter_2.values()) + 3 ** 4
demon_cv_n_counter_1 = sum(cv_no_bound_counter_1.values()) + 3 ** 2

bound_counter_2 = {k: log((v + 1) / demon_b_counter_2) for k, v in bound_counter_2.items()}
bound_counter_1 = {k: log((v + 1) / demon_b_counter_1) for k, v in bound_counter_1.items()}

no_bound_counter_2 = {k: log((v + 1) / demon_n_counter_2) for k, v in no_bound_counter_2.items()}
no_bound_counter_1 = {k: log((v + 1) / demon_n_counter_1) for k, v in no_bound_counter_1.items()}

cv_bound_counter_2 = {k: log((v + 1) / demon_cv_b_counter_2) for k, v in cv_bound_counter_2.items()}
cv_bound_counter_1 = {k: log((v + 1) / demon_cv_b_counter_1) for k, v in cv_bound_counter_1.items()}

cv_no_bound_counter_2 = {k: log((v + 1) / demon_cv_n_counter_2) for k, v in cv_no_bound_counter_2.items()}
cv_no_bound_counter_1 = {k: log((v + 1) / demon_cv_n_counter_1) for k, v in cv_no_bound_counter_1.items()}

pprint(bound_counter_2)
pprint(cv_no_bound_counter_1)

import re

b_prior = log(bound_count / (bound_count + no_bound_count))
n_prior = log(no_bound_count / (bound_count + no_bound_count))

test_gold = [d['syll'] for d in test]
test_sys = []

with open('./data/test.tsv', mode='r') as f:
    for line in f:
        _, ipa, corr = line.strip().split(sep='\t')
        ipa = ['#', ' '] + re.split(r'( )', ipa) + [' ', '#']
        corr = corr.split(sep=' ')

        parsed = []

        for i in range(2, len(ipa) - 2):
            if ipa[i] == ' ':

                b = bound_counter_2.get((ipa[i - 3], ipa[i - 1], '_', ipa[i + 1], ipa[i + 3]),
                                        log(1 / demon_b_counter_2)) + \
                    bound_counter_1.get((ipa[i - 1], '_', ipa[i + 1]), log(1 / demon_b_counter_1)) + \
                    b_prior
                n = no_bound_counter_2.get((ipa[i - 3], ipa[i - 1], '_', ipa[i + 1], ipa[i + 3]),
                                           log(1 / demon_n_counter_2)) + \
                    no_bound_counter_1.get((ipa[i - 1], '_', ipa[i + 1]), log(1 / demon_n_counter_1)) + \
                    n_prior

                # b = bound_counter_2.get((ipa[i-3], ipa[i-1], '_', ipa[i+1], ipa[i+3]), log(1/demon_b_counter_2)) +\
                #    bound_counter_1.get((ipa[i-1], '_', ipa[i+1]), log(1/demon_b_counter_1)) +\
                #    cv_bound_counter_2.get((determine_cv(ipa[i-3]), determine_cv(ipa[i-1]), '_', determine_cv(ipa[i+1]), determine_cv(ipa[i+3])), log(1/demon_cv_b_counter_2)) +\
                #    cv_bound_counter_1.get((determine_cv(ipa[i-1]), '_', determine_cv(ipa[i+1])), log(1/demon_cv_b_counter_1)) +\
                #    b_prior
                # n = no_bound_counter_2.get((ipa[i-3], ipa[i-1], '_', ipa[i+1], ipa[i+3]), log(1/demon_n_counter_2)) +\
                #    no_bound_counter_1.get((ipa[i-1], '_', ipa[i+1]), log(1/demon_n_counter_1)) +\
                #    cv_no_bound_counter_2.get((determine_cv(ipa[i-3]), determine_cv(ipa[i-1]), '_', determine_cv(ipa[i+1]), determine_cv(ipa[i+3])), log(1/demon_cv_n_counter_2)) +\
                #    cv_no_bound_counter_1.get((determine_cv(ipa[i-1]), '_', determine_cv(ipa[i+1])), log(1/demon_cv_n_counter_1)) +\
                #    n_prior

                if b > n:
                    parsed.append('.')

                # print((ipa[i-3], ipa[i-1], '_', ipa[i+1], ipa[i+3]))
                # print('b:',
                #      bound_counter_2.get((ipa[i-3], ipa[i-1], '_', ipa[i+1], ipa[i+3]), log(1/demon_b_counter_2)) +
                #      bound_counter_1.get((ipa[i-1], '_', ipa[i+1]), log(1/demon_b_counter_1)) +
                #      b_prior)
                # print('n:',
                #      no_bound_counter_2.get((ipa[i-3], ipa[i-1], '_', ipa[i+1], ipa[i+3]), log(1/demon_n_counter_2)) +
                #      no_bound_counter_1.get((ipa[i-1], '_', ipa[i+1]), log(1/demon_n_counter_1)) +
                #      n_prior)
            else:
                parsed.append(ipa[i])

        if parsed != corr:
            print('parsed:', ''.join(parsed))
            print('corr:', ''.join(corr))
            print('----------')

        test_sys.append(get_syll_indices(parsed))

evaluate(test_gold, test_sys)

from collections import Counter

uni_bndry_cntr = Counter()
uni_non_bndry_cntr = Counter()
bi_bndry_cntr = Counter()
bi_non_bndry_cntr = Counter()

with open('./data/train.tsv', mode='r') as f:
    for line in f:
        _, _, syll = line.strip().split(sep='\t')
        syll = syll.split(sep=' ')

        delimited = []
        for i, symbol in enumerate(syll[:-1]):
            if symbol != '.' and syll[i + 1] != '.':
                delimited.extend([symbol, '|'])
            else:
                delimited.append(symbol)

        delimited.append(syll[-1])
        delimited = ['#', '|'] + delimited + ['|', '#']

        for i in range(2, len(delimited) - 2):

            if delimited[i] == '|':
                uni_non_bndry_cntr[
                    (delimited[i - 1], '_', delimited[i + 1])
                ] += 1
                bi_non_bndry_cntr[
                    (delimited[i - 3], delimited[i - 1], '_', delimited[i + 1], delimited[i + 3])
                ] += 1
            elif delimited[i] == '.':
                uni_bndry_cntr[
                    (delimited[i - 1], '_', delimited[i + 1])
                ] += 1
                bi_bndry_cntr[
                    (delimited[i - 3], delimited[i - 1], '_', delimited[i + 1], delimited[i + 3])
                ] += 1


def determine_cv(ipa):
    if ipa in CONS:
        return 'C'
    elif ipa in VOWEL:
        return 'V'
    else:
        return '#'


uni_bndry_cv_cntr = Counter()
uni_non_bndry_cv_cntr = Counter()
bi_bndry_cv_cntr = Counter()
bi_non_bndry_cv_cntr = Counter()

for (pre, sep, post), freq in uni_bndry_cntr.items():
    uni_bndry_cv_cntr[(determine_cv(pre), sep, determine_cv(post))] += freq

for (pre, sep, post), freq in uni_non_bndry_cntr.items():
    uni_non_bndry_cv_cntr[(determine_cv(pre), sep, determine_cv(post))] += freq

for (prepre, pre, sep, post, postpost), freq in bi_bndry_cntr.items():
    bi_bndry_cv_cntr[(determine_cv(prepre), determine_cv(pre), sep, determine_cv(post), determine_cv(postpost))] += freq

for (prepre, pre, sep, post, postpost), freq in bi_non_bndry_cntr.items():
    bi_non_bndry_cv_cntr[
        (determine_cv(prepre), determine_cv(pre), sep, determine_cv(post), determine_cv(postpost))] += freq

s = sum(uni_non_bndry_cntr.values())
uni_non_bndry_prob = {k: v / s for k, v in uni_non_bndry_cntr.items()}

num_bndry = sum(bi_bndry_cntr.values())
num_non_bndry = sum(bi_non_bndry_cntr.values())

pri_bndry = num_bndry / (num_bndry + num_non_bndry)
pri_non_bndry = num_non_bndry / (num_bndry + num_non_bndry)


def calc_likelihood(bi_env):
    bndry_lik = bi_bndry_prob.get(bi_env, 0)
    non_bndry_lik = bi_non_bndry_prob.get(bi_env, 0)

    return bndry_lik, non_bndry_lik


def calc_likelihood(bi_env, method='bi_seg'):
    if method == 'bi_seg':
        bndry_lik = bi_bndry_prob.get(bi_env, 0)
        non_bndry_lik = bi_non_bndry_prob.get(bi_env, 0)
    elif method == 'uni_seg':
        uni_env = (bi_env[1], '_', bi_env[3])
        bndry_lik = uni_bndry_prob.get(uni_env, 0)
        non_bndry_lik = uni_non_bndry_prob.get(uni_env, 0)

    return bndry_lik, non_bndry_lik


def calc_likelihood(bi_env, method='bi_seg'):
    if method == 'bi_seg':
        bndry_lik = bi_bndry_prob.get(bi_env, 0)
        non_bndry_lik = bi_non_bndry_prob.get(bi_env, 0)
    elif method == 'uni_seg':
        uni_env = (bi_env[1], '_', bi_env[3])
        bndry_lik = uni_bndry_prob.get(uni_env, 0)
        non_bndry_lik = uni_non_bndry_prob.get(uni_env, 0)
    elif method == 'bi_cv':
        bi_env = (
        determine_cv(bi_env[0]), determine_cv(bi_env[1]), '_', determine_cv(bi_env[3]), determine_cv(bi_env[4]))
        bndry_lik = bi_bndry_cv_prob.get(bi_env, 0)
        non_bndry_lik = bi_non_bndry_cv_prob.get(bi_env, 0)
    elif method == 'uni_cv':
        uni_env = (determine_cv(bi_env[1]), '_', determine_cv(bi_env[3]))
        bndry_lik = uni_bndry_cv_prob.get(uni_env, 0)
        non_bndry_lik = uni_non_bndry_cv_prob.get(uni_env, 0)

    return bndry_lik, non_bndry_lik

def calc_likelihood(bi_env, method='bi_seg'):
    if method == 'bi_seg':
        bndry_lik = bi_bndry_prob.get(bi_env, 0)
        non_bndry_lik = bi_non_bndry_prob.get(bi_env, 0)
    elif method == 'uni_seg':
        uni_env = (bi_env[1], '_', bi_env[3])
        bndry_lik = uni_bndry_prob.get(uni_env, 0)
        non_bndry_lik = uni_non_bndry_prob.get(uni_env, 0)
    elif method == 'bi_cv':
        bi_env = (determine_cv(bi_env[0]), determine_cv(bi_env[1]), '_', determine_cv(bi_env[3]), determine_cv(bi_env[4]))
        bndry_lik = bi_bndry_cv_prob.get(bi_env, 0)
        non_bndry_lik = bi_non_bndry_cv_prob.get(bi_env, 0)
    elif method == 'uni_cv':
        uni_env = (determine_cv(bi_env[1]), '_', determine_cv(bi_env[3]))
        bndry_lik = uni_bndry_cv_prob.get(uni_env, 0)
        non_bndry_lik = uni_non_bndry_cv_prob.get(uni_env, 0)
    elif method == 'uni_bi_seg':
        uni_env = (bi_env[1], '_', bi_env[3])
        bndry_lik = bi_bndry_prob.get(bi_env, 0) * uni_bndry_prob.get(uni_env, 0)
        non_bndry_lik = bi_non_bndry_prob.get(bi_env, 0) * uni_non_bndry_prob.get(uni_env, 0)
    return bndry_lik, non_bndry_lik

def calc_likelihood(bi_env, method='bi_seg'):
    if method == 'bi_seg':
        bndry_lik = bi_bndry_prob.get(bi_env, 1/(num_bndry + 1))
        non_bndry_lik = bi_non_bndry_prob.get(bi_env, 1/(num_non_bndry + 1))
    elif method == 'uni_seg':
        uni_env = (bi_env[1], '_', bi_env[3])
        bndry_lik = uni_bndry_prob.get(uni_env, 1/(num_bndry + 1))
        non_bndry_lik = uni_non_bndry_prob.get(uni_env, 1/(num_non_bndry + 1))
    elif method == 'bi_cv':
        bi_env = (determine_cv(bi_env[0]), determine_cv(bi_env[1]), '_', determine_cv(bi_env[3]), determine_cv(bi_env[4]))
        bndry_lik = bi_bndry_cv_prob.get(bi_env, 1/(num_bndry + 1))
        non_bndry_lik = bi_non_bndry_cv_prob.get(bi_env, 1/(num_non_bndry + 1))
    elif method == 'uni_cv':
        uni_env = (determine_cv(bi_env[1]), '_', determine_cv(bi_env[3]))
        bndry_lik = uni_bndry_cv_prob.get(uni_env, 1/(num_bndry + 1))
        non_bndry_lik = uni_non_bndry_cv_prob.get(uni_env, 1/(num_non_bndry + 1))
    elif method == 'uni_bi_seg':
        uni_env = (bi_env[1], '_', bi_env[3])
        bndry_lik = bi_bndry_prob.get(bi_env, 1/(num_bndry + 1)) * uni_bndry_prob.get(uni_env, 1/(num_bndry + 1))
        non_bndry_lik = bi_non_bndry_prob.get(bi_env, 1/(num_non_bndry + 1)) * uni_non_bndry_prob.get(uni_env, 1/(num_non_bndry + 1))
    return bndry_lik, non_bndry_lik


import re

test_sys = []

with open('./data/test.tsv', mode='r') as f:
    for line in f:
        _, ipa, _ = line.strip().split(sep='\t')
        ipa = ['#', ' '] + re.split(r'( )', ipa) + [' ', '#']

        syllabified = []

        for i in range(2, len(ipa) - 2):
            if ipa[i] == ' ':
                bi_env = (ipa[i - 3], ipa[i - 1], '_', ipa[i + 1], ipa[i + 3])
                b = calc_likelihood(bi_env, method='bi_seg')[0] * pri_bndry
                n = calc_likelihood(bi_env, method='bi_seg')[1] * pri_non_bndry

                if b > n:
                    syllabified.append('.')
            else:
                syllabified.append(ipa[i])

        test_sys.append(get_syll_indices(syllabified))

test_gold = [d['syll'] for d in test]
evaluate(test_gold, test_sys)