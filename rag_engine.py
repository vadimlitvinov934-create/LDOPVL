import os
import json
import time
import numpy as np
import google.generativeai as genai

class RAG:
    def __init__(self, key):
        genai.configure(api_key=key)
        self.chat_model = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config=genai.types.GenerationConfig(temperature=0.3)
        ) 
        self.embed_model = 'models/text-embedding-004'
        
        self.db = []  
        self.limit = 0.45 
        
        # Файл, куда сохраняем "мозги" чтобы не грузить каждый раз
        self.CACHE_FILE = "vector_store.json"
        
        self.faq = {}
        self.load_faq()

    def load_faq(self):
        try:
            with open('common_answers.json', 'r', encoding='utf-8') as f:
                self.faq = json.load(f)
        except:
            self.faq = {}

    # --- КЭШИРОВАНИЕ (ДЛЯ СКОРОСТИ) ---
    def save_cache(self):
        print("💾 Сохраняю базу в файл...", end="")
        try:
            # Numpy массив нельзя сохранить в JSON, переводим в список
            serializable_db = []
            for item in self.db:
                serializable_db.append({
                    'text': item['text'],
                    'vec': item['vec'].tolist(),
                    'src': item['src']
                })
            
            with open(self.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(serializable_db, f)
            print(" Готово!")
        except Exception as e:
            print(f" Ошибка сохранения: {e}")

    def load_cache(self):
        if os.path.exists(self.CACHE_FILE):
            print(f"⚡ Найден кэш! Загружаю мгновенно...")
            try:
                with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.db = []
                for item in data:
                    # Переводим список обратно в Numpy для математики
                    item['vec'] = np.array(item['vec'])
                    self.db.append(item)
                
                print(f"✅ База восстановлена! Фрагментов: {len(self.db)}\n")
                return True 
            except Exception as e:
                print(f"⚠️ Ошибка кэша: {e}")
        return False

    def load_data(self, path):
        # 1. Пытаемся загрузить кэш (СКОРОСТЬ)
        if self.load_cache():
            return 

        # 2. Если кэша нет — сканируем файлы (ДОЛГО)
        print(f"\n📂 Кэша нет. Сканирую папку: {path}")
        if not os.path.exists(path):
            print("Папки нет!")
            return

        self.db = [] 
        files = os.listdir(path)
        
        for name in files:
            if name.endswith(".txt"):
                full_path = os.path.join(path, name)
                text = None
                for enc in ['utf-8', 'cp1251', 'windows-1251']:
                    try:
                        with open(full_path, 'r', encoding=enc) as f:
                            text = f.read()
                        break 
                    except: continue 

                if text and text.strip():
                    self.add_to_db(text, name)
                    print(f"+ {name}")
                else:
                    print(f"- {name}")
        
        # 3. После сканирования — сохраняем!
        if self.db:
            self.save_cache()
            
        print(f"Готово. В базе {len(self.db)} фрагментов.\n")

    def add_to_db(self, text, source):
        size = 1000
        chunks = [text[i:i+size] for i in range(0, len(text), size)]
        try:
            res = genai.embed_content(
                model=self.embed_model, content=chunks, task_type="retrieval_document"
            )
            for i, chunk in enumerate(chunks):
                self.db.append({
                    'text': chunk, 'vec': np.array(res['embedding'][i]), 'src': source
                })
            time.sleep(0.5)
        except Exception as e:
            print(f"Ошибка API: {e}")

    # --- ИСПРАВЛЕНИЕ ОШИБОК ---
    def fix_text(self, text):
        if len(text) < 3: return text
        prompt = f"Исправь опечатки в тексте: '{text}'. Верни ТОЛЬКО исправленный текст."
        try:
            return self.chat_model.generate_content(prompt).text.strip()
        except:
            return text

    def is_greeting(self, query):
        greetings = ['привет', 'здравствуй', 'добрый день', 'салам', 'хай']
        farewells = ['пока', 'до свидания', 'спасибо']
        words = query.lower().split()
        if len(words) < 5:
            for w in words:
                if any(g in w for g in greetings + farewells): return True
        return False

    def ai_filter(self, chunks, query):
        raw = "\n\n".join([f"Текст {i+1}: {c['text']}" for i, c in enumerate(chunks)])
        prompt = f"Проанализируй текст. Выбери ТОЛЬКО то, что нужно для ответа на: '{query}'.\nТекст:\n{raw}"
        try:
            return self.chat_model.generate_content(prompt).text
        except:
            return raw

    def ask(self, user_query):
        # 1. Исправляем ошибки
        query = self.fix_text(user_query)
        print(f"Запрос: {user_query} -> {query}") # Для отладки

        if self.is_greeting(query):
            try:
                ans = self.chat_model.generate_content(f"Ответь вежливо на приветствие: '{query}'. Ты Nexus AI.").text
                return {"ans": ans, "refs": []}
            except:
                return {"ans": "Здравствуйте!", "refs": []}

        if not self.db: return {"ans": "База пустая.", "refs": []}

        try:
            q_embed = genai.embed_content(
                model=self.embed_model, content=query, task_type="retrieval_query"
            )['embedding']
            
            vec = np.array(q_embed)
            scores = []
            for item in self.db:
                score = np.dot(vec, item['vec'])
                scores.append((score, item))

            scores.sort(key=lambda x: x[0], reverse=True)
            
            if scores[0][0] < self.limit:
                if "кто ты" in query.lower() or "что умеешь" in query.lower():
                     pass 
                else:
                    return {"ans": "В моих файлах про это ничего нет.", "refs": []}

            best = scores[:3]
            
            refs = {}
            for sc, item in best:
                name = item['src']
                perc = int(sc * 100)
                if name not in refs or perc > refs[name]['score']:
                    refs[name] = {'score': perc, 'text': item['text']}
            
            refs_list = [{"name": k, "score": v['score'], "snippet": v['text']} for k, v in refs.items()]

            facts = self.ai_filter([x[1] for x in best], query)

            final_prompt = f"""
            СИСТЕМНАЯ РОЛЬ:
            Ты — Nexus AI, умный аналитический ассистент проекта LDO.
            
            ТВОЙ ТОН:
            - Вежливый, деловой, профессиональный.
            - Обращайся к пользователю на "Вы".
            
            ЯЗЫКОВОЙ МОДУЛЬ:
            1. Отвечай СТРОГО на языке вопроса.
            
            ФАКТЫ ИЗ БАЗЫ:
            {facts}

            ВОПРОС: "{query}"
            """
            
            ans = self.chat_model.generate_content(final_prompt).text
            return {"ans": ans, "refs": refs_list}

        except Exception as e:
            return {"ans": f"Ошибка: {e}", "refs": []}