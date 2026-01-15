"""
Обертки для работы с моделями из src.models вместо external.evoprompt.llm_client
"""
from langchain_core.prompts import ChatPromptTemplate
from typing import Union, List


def _escape_braces(text: str) -> str:
    """
    Экранирует фигурные скобки в тексте для LangChain шаблонов.
    Заменяет { на {{ и } на }}, но только если это не переменные шаблона.
    """
    # Простая замена всех фигурных скобок на экранированные
    # Это безопасно, так как мы не используем переменные шаблона
    return text.replace("{", "{{").replace("}", "}}")


def llm_query(data: Union[str, List[str]], model, task: bool = False, **config) -> Union[str, List[str]]:
    """
    Обертка для llm_query из EvoPrompt, работающая с моделями из src.models
    
    Args:
        data: строка или список строк с промптами
        model: объект модели из src.models (BaseLLM)
        task: если True, берет только первую часть ответа до "\n\n"
        **config: дополнительные параметры (temperature, max_tokens и т.д.)
    
    Returns:
        str или List[str] в зависимости от типа data
    """
    temperature = config.get('temperature', 0.7)
    max_tokens = config.get('max_tokens', 1000)
    
    # Обновляем температуру модели, если нужно
    if hasattr(model, 'llm') and hasattr(model.llm, 'temperature'):
        original_temp = model.llm.temperature
        model.llm.temperature = temperature
    else:
        original_temp = None
    
    try:
        if isinstance(data, list):
            # Batch обработка
            results = []
            for prompt_text in data:
                # Экранируем фигурные скобки для LangChain
                escaped_prompt = _escape_braces(prompt_text)
                prompt_template = ChatPromptTemplate.from_messages([
                    ("user", escaped_prompt)
                ])
                response = model.generate(prompt_template)
                result = response.content if hasattr(response, 'content') else str(response)
                
                if task:
                    result = result.strip().split("\n\n")[0]
                else:
                    result = result.strip()
                
                results.append(result)
            return results
        else:
            # Одиночный запрос
            # Экранируем фигурные скобки для LangChain
            escaped_data = _escape_braces(data)
            prompt_template = ChatPromptTemplate.from_messages([
                ("user", escaped_data)
            ])
            response = model.generate(prompt_template)
            result = response.content if hasattr(response, 'content') else str(response)
            
            if task:
                result = result.strip().split("\n\n")[0]
            else:
                result = result.strip()
            
            return result
    finally:
        # Восстанавливаем оригинальную температуру
        if original_temp is not None and hasattr(model, 'llm'):
            model.llm.temperature = original_temp


def paraphrase(sentence: Union[str, List[str]], model, **kwargs) -> Union[str, List[str]]:
    """
    Обертка для paraphrase из EvoPrompt, работающая с моделями из src.models
    
    Args:
        sentence: строка или список строк для парафраза
        model: объект модели из src.models (BaseLLM)
        **kwargs: дополнительные параметры
    
    Returns:
        str или List[str] в зависимости от типа sentence
    """
    if isinstance(sentence, list):
        resample_templates = [
            f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{s}\nOutput:"
            for s in sentence
        ]
    else:
        resample_templates = f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{sentence}\nOutput:"
    
    results = llm_query(resample_templates, model, task=False, **kwargs)
    return results
