"""Lightweight, offline candidate search for incomplete FAQ duplicates."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix

from faq_report_rebuild import context, norm, differences


TOKEN = re.compile(r'[^\W\d_]+|\d+(?:[.,:/-]\d+)*', re.UNICODE)
LIST_NUMBER = re.compile(r'(?m)^\s*\d{1,2}[.)]\s+')
RISK_NUMBER = re.compile(
    r'(?<!\w)(\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?)\s*'
    r'(%|рабоч\w*\s+(?:дн\w*|день)|календар\w*\s+(?:дн\w*|день)|дн(?:я|ей)?\b|день\b|'
    r'руб\w*|₽|тыс\w*|млн\w*|мес\w*|год\w*|лет\b|час\w*)', re.I)
ARTICLE = re.compile(r'\b(статья|пункт|подпункт)\s*(\d+(?:\.\d+)*)', re.I)
QUOTES = re.compile(r'[«"]([^»"]+)[»"]')
NAMED_DEPOSIT = re.compile(r'\b(?:вклад|сч[её]т)\w*\s*[«"]([^»"]+)[»"]', re.I)
PRIME = re.compile(r'\bсберпрайм\b(?:\s*(?:\+|плюс))?', re.I)
SCOPE = re.compile(r'\b(?:сберперв\w*|сберпремьер\w*|доступн\w* выгод\w*|свой максимум|'
                   r'подушка безопасности|страхование по часам|дмс|защита от травм)\b')
URL = re.compile(r'https?://\S+', re.I)
QUOTE_CHARS = str.maketrans('', '', '«»„“”"')
DASH_CHARS = str.maketrans({'—': '–', '−': '–'})
OPERATIONS = [
    (r'\b(?:внос\w*|внести|полож(?!ен)\w*|пополн\w*)\b', r'\b(?:снят\w*|сним\w*|выдал|выдач\w*)\b', 'Внесение и выдача'),
    (r'\bподключ\w*\b', r'\bотключ\w*\b', 'Подключение и отключение'),
    (r'\bдобав\w*\b', r'\bудал\w*\b', 'Добавление и удаление'),
    (r'\bоткр\w*\b', r'\bзакр\w*\b', 'Открытие и закрытие'),
]
OPERATIONS = [(re.compile(a), re.compile(b), label) for a, b, label in OPERATIONS]
STRONG_NEGATION = re.compile(r'\b(?:нельзя|запрещ\w*|не\s+допуска\w*|не\s+предусмотр\w*)\b')


@dataclass(frozen=True)
class Thresholds:
    answer: float = .76
    question: float = .46
    coverage: float = .93
    containment_question: float = .20
    embedding: float | None = None
    strong_answer: float = .95
    same_question_answer: float = .44


def words(text):
    return TOKEN.findall(LIST_NUMBER.sub('', text).casefold().replace('ё', 'е'))


def answer_key(text):
    # Preserve units, +, digits, punctuation inside numbers, and case-sensitive URLs.
    if re.search(r'https?://', text, re.I):
        return norm(text)
    normalized = unicodedata.normalize('NFC', text).translate(QUOTE_CHARS).translate(DASH_CHARS)
    return norm(normalized.casefold().replace('ё', 'е')).removesuffix('.').rstrip()


def weighted_matrix(documents, max_df=.20):
    """Return unit TF-IDF rows for unigrams and two-word phrases."""
    count = len(documents)
    terms = [Counter([*tokens, *zip(tokens, tokens[1:])]) for tokens in documents]
    frequency = Counter(term for row in terms for term in row)
    vocab = {term: k for k, term in enumerate(
        term for term, df in frequency.items() if df <= max(2, count * max_df))}
    indptr, indices, data = [0], [], []
    for row in terms:
        weighted = [(vocab[t], (1 + math.log(n)) * (1 + math.log((1 + count) / (1 + frequency[t]))))
                    for t, n in row.items() if t in vocab]
        size = math.sqrt(sum(w * w for _, w in weighted))
        indices.extend(i for i, _ in weighted)
        data.extend(w / size for _, w in weighted)
        indptr.append(len(data))
    result = csr_matrix((np.asarray(data, dtype=np.float32), indices, indptr),
                        shape=(count, len(vocab)), dtype=np.float32)
    result.sort_indices()
    return result


class Corpus:
    def __init__(self, records):
        self.cards = [records[key] for key in sorted(records)]
        self.index = {card.id: i for i, card in enumerate(self.cards)}
        self.form = [answer_key(c.answer) for c in self.cards]
        answers = [words(c.answer) for c in self.cards]
        questions = [words(c.question) for c in self.cards]
        self.answer_matrix = weighted_matrix(answers)
        self.question_matrix = weighted_matrix(questions, max_df=.40)
        counts = Counter(w for tokens in answers for w in set(tokens))
        self.idf = {w: 1 + math.log((1 + len(answers)) / (1 + n)) for w, n in counts.items()}
        self.tokens = [Counter(tokens) for tokens in answers]
        self.token_text = [' ' + ' '.join(tokens) + ' ' for tokens in answers]
        self.token_count = list(map(len, answers))
        self.weight = [sum(self.idf[w] * n for w, n in ts.items()) for ts in self.tokens]
        self.question_forms = [{norm(q).casefold() for q in (c.question, *c.alternatives)}
                               for c in self.cards]
        self.product = np.array([c.product_key[0] for c in self.cards], dtype=object)

    def pool(self, top_k=12, block=96, answer_floor=.36, question_floor=.58, dense=None,
             dense_search_floor=None):
        """Keep independent neighbour budgets inside and across products for each signal."""
        if top_k < 1 or block < 1:
            raise ValueError('Размер блока и число соседей должны быть положительными.')
        seen = defaultdict(set)

        def select(row, columns, scores, tag):
            if not columns.size:
                return
            eligible = np.array([j != row and self.form[j] != self.form[row] for j in columns])
            same = (self.product[columns] == self.product[row]) & bool(self.product[row])
            for mask in (eligible & same, eligible & ~same):
                offsets = np.nonzero(mask)[0]
                if offsets.size > top_k:
                    order = np.lexsort((columns[offsets], -scores[offsets]))[:top_k]
                    offsets = offsets[order]
                for other in columns[offsets]:
                    seen[tuple(sorted((row, int(other))))].add(tag)

        for matrix, floor, tag in ((self.answer_matrix, answer_floor, 'ответы'),
                                   (self.question_matrix, question_floor, 'вопросы')):
            transpose = matrix.T
            for start in range(0, len(self.cards), block):
                batch = (matrix[start:start + block] @ transpose).tocsr()
                for offset in range(batch.shape[0]):
                    lo, hi = batch.indptr[offset:offset + 2]
                    columns, scores = batch.indices[lo:hi], batch.data[lo:hi]
                    good = scores >= floor
                    select(start + offset, columns[good], scores[good], tag)

        # Named question overlaps, including alternatives, rescue some paraphrases with low lexical cosine.
        inverted = defaultdict(list)
        for row, forms in enumerate(self.question_forms):
            for form in forms:
                if len(words(form)) >= 3:
                    inverted[form].append(row)
        for rows in inverted.values():
            if len(rows) > 24:
                continue
            for n, i in enumerate(rows):
                for j in rows[n + 1:]:
                    if self.form[i] != self.form[j]:
                        seen[i, j].add('та же формулировка вопроса')

        if dense_search_floor is not None:
            if dense is None:
                raise ValueError('Для поиска соседей E5 нужен проверенный кэш.')
            vectors, positions = dense
            keys = np.array(sorted(positions))
            if keys.size:
                sample = vectors[[positions[int(i)] for i in keys]]
                for start in range(0, len(keys), block):
                    scores = sample[start:start + block] @ sample.T
                    for offset, score in enumerate(scores):
                        keep = score >= dense_search_floor
                        select(int(keys[start + offset]), keys[keep], score[keep], 'E5')
        return seen

    def features(self, pairs, dense=None):
        keys = sorted(pairs)
        for start in range(0, len(keys), 1024):
            batch = keys[start:start + 1024]
            ai, bi = np.asarray(batch).T
            score_a = np.asarray(self.answer_matrix[ai].multiply(self.answer_matrix[bi]).sum(axis=1)).ravel()
            score_q = np.asarray(self.question_matrix[ai].multiply(self.question_matrix[bi]).sum(axis=1)).ravel()
            for (i, j), a_score, q_score in zip(batch, score_a, score_q):
                a, b = self.cards[i], self.cards[j]
                shared = sum(self.idf[w] * n for w, n in (self.tokens[i] & self.tokens[j]).items())
                cov_a = shared / self.weight[i] if self.weight[i] else 0
                cov_b = shared / self.weight[j] if self.weight[j] else 0
                same_question = bool(self.question_forms[i] & self.question_forms[j])
                dense_score = None
                if dense is not None:
                    v, pos = dense
                    if i in pos and j in pos:
                        dense_score = float(np.clip(v[pos[i]] @ v[pos[j]], -1, 1))
                score = {
                    'a': a.id, 'b': b.id, 'answer': min(1., float(a_score)),
                    'question': min(1., float(q_score)), 'same_question': same_question,
                    'cover_a': min(1., cov_a), 'cover_b': min(1., cov_b), 'embedding': dense_score,
                    'literal': min(self.token_count[i], self.token_count[j]) >= 8
                    and (self.token_text[i] in self.token_text[j] or self.token_text[j] in self.token_text[i]),
                }
                yield (i, j), score


def _numbers_with_units(text):
    result = defaultdict(set)
    for value, unit in RISK_NUMBER.findall(text.casefold()):
        unit = norm(unit)
        if unit.startswith(('рабоч', 'календар')):
            unit = unit.split()[0][:5] + ' дни'
        elif unit.startswith(('дн', 'ден')):
            unit = 'дни'
        elif unit == 'лет':
            unit = 'год'
        else:
            unit = unit[:3]
        result[unit].add(float(value.replace(' ', '').replace('\u00a0', '').replace(',', '.')))
    return result


def risks(a, b):
    relation, subtype = context(a, b)
    qa, qb = [norm(x.question).casefold().replace('ё', 'е') for x in (a, b)]
    qa_refs = {(k.casefold(), n) for k, n in ARTICLE.findall(qa)}
    qb_refs = {(k.casefold(), n) for k, n in ARTICLE.findall(qb)}
    hard, caution = [], []
    same_product = bool(a.product_key[0] and a.product_key[0] == b.product_key[0])
    if relation == 'different' and 'внутри' in subtype:
        hard.append(subtype)
    elif relation != 'same':
        caution.append(subtype)
    if any({n for k2, n in qa_refs if k2 == k}.isdisjoint({n for k2, n in qb_refs if k2 == k})
           for k in set(k for k, _ in qa_refs) & set(k for k, _ in qb_refs)):
        hard.append('Разные статьи или пункты')
    names_a = {norm(s) for s in QUOTES.findall(qa)}
    names_b = {norm(s) for s in QUOTES.findall(qb)}
    if same_product and a.product_key[0] == 'вклады и счета':
        names_a = names_a or {norm(s) for s in NAMED_DEPOSIT.findall(a.answer[:160].casefold())}
        names_b = names_b or {norm(s) for s in NAMED_DEPOSIT.findall(b.answer[:160].casefold())}
    prime_a, prime_b = PRIME.search(qa), PRIME.search(qb)
    if prime_a and prime_b and (prime_a[0] == 'сберпрайм') != (prime_b[0] == 'сберпрайм'):
        hard.append('Разные пакеты СберПрайм')
    scope_a, scope_b = set(SCOPE.findall(qa)), set(SCOPE.findall(qb))
    if scope_a and scope_b and scope_a.isdisjoint(scope_b):
        caution.append('Отличается явно названная программа или пакет')
    named_scope = a.product_key[0] in ('вклады и счета', 'страхование', 'инвестиции и ценные бумаги ук')
    if same_product and named_scope and names_a and names_b and names_a.isdisjoint(names_b):
        hard.append('Разные явные названия продукта или программы')
    for x, y, reason in OPERATIONS:
        a_action, b_action = (bool(x.search(qa)), bool(y.search(qa))), (bool(x.search(qb)), bool(y.search(qb)))
        if (a_action, b_action) in (((True, False), (False, True)), ((False, True), (True, False))):
            hard.append('Разные операции: ' + reason)
    na, nb = _numbers_with_units(a.answer), _numbers_with_units(b.answer)
    for unit in sorted(na.keys() & nb.keys()):
        if na[unit].isdisjoint(nb[unit]):
            hard.append(f'Разные значения ({unit}): {sorted(na[unit])} ↔ {sorted(nb[unit])}')
        elif na[unit] != nb[unit]:
            caution.append(f'Дополнительные числовые условия ({unit})')
    if bool(STRONG_NEGATION.search(a.answer.casefold())) != bool(STRONG_NEGATION.search(b.answer.casefold())):
        caution.append('Явный запрет только в одном ответе')
    if (set(URL.findall(a.answer)) != set(URL.findall(b.answer))
            or any(flag.startswith('Ссылка') for flag in differences(a, b))):
        caution.append('Разные ссылки')
    return same_product, relation, hard, caution


def qualify(feature, flags, cut, same_product_only=False):
    same, relation, hard, caution = flags
    if same_product_only and not same:
        return 'вне продукта', ''
    question = feature['question']
    answer = feature['answer']
    exact_q = feature['same_question']
    containment = max(feature['cover_a'], feature['cover_b'])
    lexical = answer >= cut.answer and (question >= cut.question or exact_q)
    same_form = exact_q and answer >= cut.same_question_answer
    strong = answer >= cut.strong_answer
    literal = feature.get('literal', False) and answer >= .45
    covers = containment >= cut.coverage and answer >= .23 and (question >= cut.containment_question or exact_q)
    embedding = (cut.embedding is not None and feature['embedding'] is not None
                 and feature['embedding'] >= cut.embedding
                 and (question >= cut.question or exact_q or answer >= cut.answer))
    if not (lexical or same_form or strong or covers or embedding or literal):
        return 'ниже порога', ''
    signals = [(lexical, 'TF-IDF вопроса и ответа'), (same_form, 'общий вопрос'),
               (strong, 'высокий cosine ответа'), (covers, 'покрытие'),
               (literal, 'текстовая вложенность'), (embedding, 'E5')]
    method = ', '.join(name for active, name in signals if active)
    if hard:
        return 'существенное различие', method
    strong_question = ((exact_q and answer >= .65) or (question >= .65 and answer >= .78)
                       or (question >= .58 and containment >= .96))
    if caution or relation != 'same' or not strong_question:
        return 'сверка контекста', method
    return 'приоритетная вычитка', method


def load_embedding_cache(path, snapshot, corpus):
    """Match row positions from the original run to current cards by ID and answer text."""
    import pandas as pd

    if Path(snapshot).suffix.lower() == '.csv':
        frame = pd.read_csv(snapshot, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    elif Path(snapshot).suffix.lower() == '.pkl':
        frame = pd.read_pickle(snapshot)
    else:
        raise ValueError('Доступны CSV со столбцами Идентификатор и Ответ или доверенный локальный PKL.')
    ids = frame['Идентификатор'].fillna('').astype(str).str.strip()
    answers = frame['Ответ'].fillna('').astype(str).str.strip()
    if ids.empty or ids.eq('').any() or ids.duplicated().any() or answers.eq('').any():
        raise ValueError('В снимке кэша пустой ответ или пустой/повторный ID.')
    vectors = np.load(path, mmap_mode='r', allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] < 1 or vectors.dtype.kind != 'f':
        raise ValueError('Размерность или тип NPY не соответствует снимку.')
    matches = [(i, corpus.index[k]) for i, (k, a) in enumerate(zip(ids, answers))
               if k in corpus.index and corpus.cards[corpus.index[k]].answer.strip() == a]
    if not matches:
        raise ValueError('Ни одного совпадения ID и ответа между выгрузками и снимком.')
    a, b = np.asarray(matches).T
    selected = vectors[a].astype(np.float32)
    lengths = np.linalg.norm(selected, axis=1)
    if not np.isfinite(lengths).all() or (lengths == 0).any():
        raise ValueError('Кэш содержит нулевые или нечисловые векторы среди совпавших FAQ.')
    selected /= lengths[:, None]
    return selected, dict(zip(b.tolist(), range(len(b))))
