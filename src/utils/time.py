def format_time(seconds):
    """
    Форматирует время в секундах в читаемый формат.
    
    Вход:
        seconds (float): время в секундах
    Выход:
        str: отформатированная строка времени (мм:сс или чч:мм:сс)
    """
    if seconds < 60:
        return f"{seconds:.1f}с"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}м {secs:.1f}с"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}ч {minutes}м {secs:.1f}с"