import time


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


class TimeEstimator:
    """
    Класс для оценки оставшегося времени на основе истории выполнения задач.
    """
    def __init__(self, total_items=None):
        """
        Инициализация планировщика времени.
        
        Вход:
            total_items (int, optional): общее количество элементов для обработки
        """
        self.total_items = total_items
        self.start_time = None
        self.item_times = []  # История времени выполнения каждого элемента
        self.current_item_start = None
        
    def start(self):
        """Запускает отслеживание времени."""
        self.start_time = time.time()
        
    def start_item(self):
        """Начинает отслеживание времени для текущего элемента."""
        self.current_item_start = time.time()
        
    def finish_item(self):
        """Завершает отслеживание времени для текущего элемента."""
        if self.current_item_start is not None:
            elapsed = time.time() - self.current_item_start
            self.item_times.append(elapsed)
            self.current_item_start = None
            return elapsed
        return 0.0
    
    def get_elapsed(self):
        """Возвращает общее прошедшее время."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def estimate_remaining(self, completed_items=None):
        """
        Оценивает оставшееся время на основе средней скорости выполнения.
        
        Вход:
            completed_items (int, optional): количество завершённых элементов
                                              (если None, используется len(self.item_times))
        Выход:
            float: оценка оставшегося времени в секундах, или None если недостаточно данных
        """
        if completed_items is None:
            completed_items = len(self.item_times)
            
        if self.total_items is None or completed_items == 0:
            return None
            
        if len(self.item_times) == 0:
            return None
            
        # Используем среднее время выполнения для оценки
        avg_time_per_item = sum(self.item_times) / len(self.item_times)
        remaining_items = self.total_items - completed_items
        
        if remaining_items <= 0:
            return 0.0
            
        return avg_time_per_item * remaining_items
    
    def get_progress_info(self, completed_items=None):
        """
        Возвращает строку с информацией о прогрессе и оставшемся времени.
        
        Вход:
            completed_items (int, optional): количество завершённых элементов
        Выход:
            str: строка с информацией о прогрессе
        """
        if completed_items is None:
            completed_items = len(self.item_times)
            
        elapsed = self.get_elapsed()
        elapsed_str = format_time(elapsed)
        
        if self.total_items is None:
            return f"⏱️  Прошло времени: {elapsed_str}"
        
        progress_pct = (completed_items / self.total_items * 100) if self.total_items > 0 else 0
        remaining = self.estimate_remaining(completed_items)
        
        info_parts = [
            f"⏱️  Прогресс: {completed_items}/{self.total_items} ({progress_pct:.1f}%)",
            f"Прошло: {elapsed_str}"
        ]
        
        if remaining is not None and remaining > 0:
            remaining_str = format_time(remaining)
            info_parts.append(f"Осталось: ~{remaining_str}")
        
        return " | ".join(info_parts)