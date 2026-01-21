import json
import re

_QUESTION_KEY_PATTERN = re.compile(r'question[_ ]?id', re.IGNORECASE)


def _validate_and_convert(data):
    """
    Валидирует и конвертирует список ответов в словарь.
    
    Вход:
        data (list[dict]): список словарей с ключами 'question_id' (int, 1-120) 
                          и 'answer' (int, 1-5)
    Выход:
        dict[int, int] или None: словарь {question_id: answer} при успехе, 
                                 None при ошибке валидации или дубликатах
    """
    if not isinstance(data, list):
        return None
    
    result = {}
    seen_ids = set()
    
    for item in data:
        if not isinstance(item, dict):
            return None
        q_id = item.get("question_id")
        answer = item.get("answer")
        
        if not isinstance(q_id, int) or not isinstance(answer, int):
            return None
        if not (1 <= q_id <= 120) or not (1 <= answer <= 5):
            return None
        if q_id in seen_ids:
            return None  # дубли
        seen_ids.add(q_id)
        result[q_id] = answer
    
    return result

def parse_response(response_content):
    """
    Парсит текстовый ответ модели и извлекает ответы на вопросы.
    
    Вход:
        response_content (str): текстовый ответ модели, содержащий JSON с ответами
    Выход:
        dict[int, int] или None: словарь {question_id: answer} при успешном парсинге,
                                 None если ответ не удалось распарсить
    """
    if not response_content:
        return None
    
    # Базовая очистка: убираем лишнее форматирование и комментарии
    def _normalize(text):
        # убираем обрамление ```json ... ```
        text = re.sub(r'```(?:json)?', '', text)
        text = text.replace('```', '')
        # убираем комментарии вида "# ..." или "// ..."
        text = re.sub(r'#.*', '', text)
        text = re.sub(r'//.*', '', text)
        # унифицируем вариант ключа question_id
        text = _QUESTION_KEY_PATTERN.sub('question_id', text)
        return text.strip()

    content = _normalize(response_content)
    
    # Шаг 1: Пытаемся прямой json.loads
    try:
        data = json.loads(content)
        return _validate_and_convert(data)
    except json.JSONDecodeError:
        pass
    
    # Шаг 2: Ищем JSON-массив внутри текста
    json_match = re.search(r'\[.*\]', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            data = json.loads(_normalize(json_str))
            return _validate_and_convert(data)
        except json.JSONDecodeError:
            pass

    # Шаг 3: Парсим список словарей без обрамляющего массива/с лишними символами
    # Формат вида: { "question_id": 1, "answer": 3 }, {"question_id": 2, "answer": 4}, ...
    pair_pattern = re.compile(
        r'"?question_id"?\s*[:=]\s*(\d+)[^{}]*?"?answer"?\s*[:=]\s*(\d+)',
        re.IGNORECASE | re.DOTALL,
    )
    pairs = pair_pattern.findall(content)
    if pairs:
        data = [{'question_id': int(q), 'answer': int(a)} for q, a in pairs]
        validated = _validate_and_convert(data)
        if validated is not None:
            return validated

    # Шаг 4: Парсим строковый список вида "48. 5" (по одной паре на строку)
    line_pattern = re.compile(
        r'^\s*(\d{1,3})\s*[\.\-:)\s]\s*([1-5])\s*$',
        re.MULTILINE,
    )
    pairs = line_pattern.findall(content)
    if pairs:
        data = [{'question_id': int(q), 'answer': int(a)} for q, a in pairs]
        validated = _validate_and_convert(data)
        if validated is not None:
            return validated

    print(f"⚠️  Ошибка: не удалось распарсить ответ модели как JSON")
    print(f"Первые 500 символов ответа:\n{content[:500]}...\n")
    return None
