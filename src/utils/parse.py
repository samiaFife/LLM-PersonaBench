import json
import re

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
    
    content = response_content.strip()
    
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
            data = json.loads(json_str)
            return _validate_and_convert(data)
        except json.JSONDecodeError:
            pass

    print(f"⚠️  Ошибка: не удалось распарсить ответ модели как JSON")
    print(f"Первые 500 символов ответа:\n{content[:500]}...\n")
    return None
