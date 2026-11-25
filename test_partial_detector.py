from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from faq_report_rebuild import FAQ
from faq_partial_detector import Corpus, Thresholds, answer_key, load_embedding_cache, qualify, risks
from faq_partial_outputs import fingerprint, search


def card(ident, title, answer, product='Карты', subproduct='Дебетовые'):
    return FAQ(ident, title, answer, (), 'main', {'Продукт': product, 'Подпродукт': subproduct})


class PartialDetectorTests(TestCase):
    def test_safe_exact_normalization_preserves_numbers_plus_and_urls(self):
        self.assertEqual(answer_key('По счёту «Удача».'), answer_key('По счету "Удача"'))
        self.assertNotEqual(answer_key('Программа Плюс +'), answer_key('Программа Плюс'))
        self.assertNotEqual(answer_key('Ставка 1,5%'), answer_key('Ставка 15%'))
        self.assertNotEqual(answer_key('https://example.test/A'), answer_key('https://example.test/a'))

    def test_cross_product_stays_in_context_review(self):
        a = card('a', 'Как перевести деньги в другой банк', 'Зайдите в приложение и откройте платежи')
        b = card('b', a.question, 'Зайдите в приложение и откройте раздел платежи', 'Переводы')
        flags = risks(a, b)
        f = dict(answer=.95, question=1, same_question=True, cover_a=.9, cover_b=.9, embedding=None)
        self.assertEqual(qualify(f, flags, Thresholds())[0], 'сверка контекста')
        self.assertEqual(qualify(f, flags, Thresholds(), same_product_only=True)[0], 'вне продукта')
        corpus = Corpus({'a': a, 'b': b})
        self.assertIn((0, 1), corpus.pool(top_k=1))

    def test_material_amounts_and_opposed_actions(self):
        a = card('a', 'Как внести наличные', 'Проверка занимает до 15 рабочих дней')
        b = card('b', 'Как снять наличные', 'Проверка занимает до 30 рабочих дней')
        hard = risks(a, b)[2]
        self.assertTrue(any('Внесение и выдача' in reason for reason in hard))
        self.assertTrue(any('дни' in reason for reason in hard))

    def test_identical_two_products_only_in_full_tab(self):
        a = card('a', 'Где посмотреть договор', 'Откройте «Договоры»', 'Кредиты')
        b = card('b', 'Что такое договор', 'Откройте "Договоры".', 'Карты')
        corpus = Corpus({'a': a, 'b': b})
        self.assertFalse(corpus.pool())

    def test_literal_containment_kept_with_low_question_score(self):
        f = dict(answer=.62, question=.03, same_question=False, cover_a=1, cover_b=.51,
                 literal=True, embedding=None)
        state, signals = qualify(f, (True, 'same', [], []), Thresholds())
        self.assertEqual(state, 'сверка контекста')
        self.assertIn('текстовая вложенность', signals)

    def test_ready_vectors_align_by_id_and_unchanged_answer(self):
        a = card('a', 'Как открыть', 'Короткая инструкция')
        b = card('b', 'Как закрыть', 'Изменённая инструкция')
        corpus = Corpus({'b': b, 'a': a})
        with TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root/'v.npy', np.array([[0, 1], [1, 0]], dtype=np.float32))
            (root/'s.csv').write_text('Идентификатор,Ответ\nb,Прежний ответ\na,Короткая инструкция\n',
                                      encoding='utf-8')
            vectors, pos = load_embedding_cache(root/'v.npy', root/'s.csv', corpus)
            self.assertEqual(pos, {0: 0})
            np.testing.assert_array_equal(vectors, [[1, 0]])

    def test_audit_excludes_known_false_from_review_and_keeps_Model_features(self):
        a = card('a', 'Как найти расходы', 'Зайдите в историю платежей через приложение')
        b = card('b', 'Как найти расходы', 'В приложении зайдите в историю платежей')
        corpus = Corpus({'a': a, 'b': b})
        bench = {(0, 1): {'decision': 'разные'}}
        found, measured = search(corpus, {(0, 1): {'вопросы'}}, bench, Thresholds(same_question_answer=.1))
        self.assertEqual(found[0]['queue'], 'существенное различие')
        self.assertNotEqual(measured[0, 1]['queue'], 'существенное различие')
        self.assertNotEqual(fingerprint(a), fingerprint(b))
