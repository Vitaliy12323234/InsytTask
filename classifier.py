import pandas as pd
import re
import json
import requests
import time

# ==========================================
# НАСТРОЙКИ
# ==========================================
MODEL_NAME = "qwen3.5:9b" 
# Используем /api/chat вместо /api/generate
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"

CATEGORIES = {
    "Авторизация": "Технический",
    "Ошибка загрузки": "Технический",
    "Организационные вопросы по обучению": "Организационный",
    "Ошибки в контенте/методологии": "Организационный",
    "Не завершается обучение": "Организационный"
}

def check_manual_review(text):
    if pd.isna(text):
        return True
    text = str(text).strip()
    if len(text) == 0:
        return True
    if len(text) < 5:
        return True
    if not re.search(r'[А-Яа-яЁё]', text):
        return True
    emotional_patterns = [
        r'^(спасибо|благодарю|ура|круто|супер|отлично|плохо|ужас|кошмар|привет|здравствуйте)[.!?\s]*$',
        r'^кошмар[!?\s]*$',
        r'^безобразие[!?\s]*$',
        r'^очень плохо[!?\s]*$',
    ]
    for pattern in emotional_patterns:
        if re.match(pattern, text.lower()):
            return True
    return False

def classify_with_ollama(text):
    system_prompt = """Ты — эксперт техподдержки образовательной платформы (LMS).
Классифицируй обращение пользователя строго в ОДНУ из 5 категорий:
1. "Авторизация" — вход, пароль, восстановление доступа, активация учётной записи.
2. "Ошибка загрузки" — сбои открытия, передачи и воспроизведения страниц, файлов, видео.
3. "Организационные вопросы по обучению" — назначение/отмена курсов, зачисление в потоки, расписание, сроки.
4. "Ошибки в контенте/методологии" — опечатки, неверные формулы, противоречия материалов, ошибки в условиях заданий.
5. "Не завершается обучение" — не сохраняется прогресс, не засчитываются результаты, не выдаётся документ при выполненных условиях.

Ответь СТРОГО в формате JSON: {"category": "название категории"}. Никакого другого текста. /no_think"""

    # Формат запроса для /api/chat
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Текст обращения: "{text}"'}
        ],
        "stream": False,
        "format": "json",
        "think": False,  # Отключаем режим размышлений
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
        
        if response.status_code != 200:
            print(f"  [!] HTTP {response.status_code}: {response.text[:200]}")
            return "Не определено", "Требует ручной обработки"
        
        result_data = response.json()
        
        if "error" in result_data:
            print(f"  [!] ОШИБКА OLLAMA: {result_data['error']}")
            return "Не определено", "Требует ручной обработки"
        
        # Извлекаем контент из ответа chat API
        message = result_data.get("message", {})
        result_text = message.get("content", "").strip()
        
        if not result_text:
            print(f"  [!] ПУСТОЙ ОТВЕТ. Полный JSON: {result_data}")
            return "Не определено", "Требует ручной обработки"
        
        result_text = re.sub(r'^```json\s*|```\s*$', '', result_text, flags=re.MULTILINE).strip()
        
        try:
            parsed = json.loads(result_text)
        except json.JSONDecodeError:
            print(f"  [!] Модель вернула не-JSON: '{result_text}'")
            return "Не определено", "Требует ручной обработки"
        
        category = parsed.get("category")
        
        if category in CATEGORIES:
            return category, CATEGORIES[category]
        else:
            print(f"  [!] Неизвестная категория: '{category}'")
            return "Не определено", "Требует ручной обработки"
            
    except requests.exceptions.ConnectionError:
        print(f"  [!] НЕТ СОЕДИНЕНИЯ с Ollama.")
        return "Не определено", "Требует ручной обработки"
    except requests.exceptions.Timeout:
        print(f"  [!] Таймаут 180 сек.")
        return "Не определено", "Требует ручной обработки"
    except Exception as e:
        print(f"  [!] Ошибка: {type(e).__name__}: {e}")
        return "Не определено", "Требует ручной обработки"

def main():
    print("Загрузка данных из tikets.csv...")
    try:
        df = pd.read_csv("tikets.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        print("ОШИБКА: Файл tikets.csv не найден!")
        return

    results = []
    total = len(df)
    print(f"Начинаем обработку {total} записей. Модель: {MODEL_NAME}")
    
    # Быстрый тест перед основной обработкой
    print("\n>>> Тест соединения с моделью...")
    test_cat, _ = classify_with_ollama("Не приходит пароль на почту")
    print(f">>> Тест: '{test_cat}'\n")
    if test_cat == "Не определено":
        print(">>> ВНИМАНИЕ: модель не отвечает корректно. Исправьте проблему и перезапустите.")
        return
    
    for index, row in df.iterrows():
        original_id = row['id']
        original_text = row['text']
        
        if check_manual_review(original_text):
            category = "Не определено"
            tag = "Требует ручной обработки"
            requires_manual = True
            print(f"[{index+1}/{total}] Ручная обработка: {str(original_text)[:40]}")
        else:
            print(f"[{index+1}/{total}] LLM: {str(original_text)[:40]}...")
            category, tag = classify_with_ollama(original_text)
            requires_manual = False
            time.sleep(0.05)

        results.append({
            "id": original_id,
            "text": original_text,
            "category": category,
            "tag": tag,
            "requires_manual_review": str(requires_manual).lower(),
            "status": "ok",
            "error": ""
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv("result.csv", index=False, encoding="utf-8")
    print("\n===========================================")
    print("ГОТОВО! Результат сохранен в файл 'result.csv'")
    print("===========================================")

if __name__ == "__main__":
    main()